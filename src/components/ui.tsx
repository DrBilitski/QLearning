import { ReactNode, useEffect, useRef, useState } from 'react';
import { fmtQ } from '../engine/format';

/**
 * Background color for a Q-value: red for unfavourable, green for favourable,
 * intensity grows with magnitude (saturating around ±10).
 */
export function valueColor(v: number, alpha = 1): string {
  const i = Math.min(1, Math.abs(v) / 10) ** 0.55;
  if (Math.abs(v) < 1e-9) return `rgba(237, 233, 254, ${alpha})`;
  if (v > 0) return `hsla(${150 - i * 10}, ${55 + i * 20}%, ${94 - i * 48}%, ${alpha})`;
  return `hsla(${358}, ${70 + i * 15}%, ${95 - i * 45}%, ${alpha})`;
}

export function valueTextColor(v: number): string {
  const i = Math.min(1, Math.abs(v) / 10) ** 0.55;
  return i > 0.55 ? '#ffffff' : '#2e1065';
}

export function Panel(props: { title: string; icon?: string; className?: string; right?: ReactNode; children: ReactNode }) {
  return (
    <section className={`panel ${props.className ?? ''}`}>
      <header className="panel-title">
        <span>
          {props.icon && <span className="panel-icon">{props.icon}</span>}
          {props.title}
        </span>
        {props.right}
      </header>
      <div className="panel-body">{props.children}</div>
    </section>
  );
}

/** A number that tweens smoothly to its new value and flashes when it changes. */
export function AnimatedNumber({
  value,
  duration,
  initial,
  format = fmtQ,
}: {
  value: number;
  duration: number;
  /** Start the first tween from this value (e.g. the old Q-value). */
  initial?: number;
  format?: (n: number) => string;
}) {
  const start = initial ?? value;
  const [shown, setShown] = useState(start);
  const [flash, setFlash] = useState(0);
  const from = useRef(start);
  const shownRef = useRef(start);

  useEffect(() => {
    if (value === shownRef.current) return;
    from.current = shownRef.current;
    setFlash((f) => f + 1);
    if (duration < 30) {
      shownRef.current = value;
      setShown(value);
      return;
    }
    const t0 = performance.now();
    let raf = 0;
    const tickFn = (t: number) => {
      const k = Math.min(1, (t - t0) / duration);
      const e = 1 - (1 - k) ** 3;
      const v = from.current + (value - from.current) * e;
      shownRef.current = k >= 1 ? value : v;
      setShown(shownRef.current);
      if (k < 1) raf = requestAnimationFrame(tickFn);
    };
    raf = requestAnimationFrame(tickFn);
    // On cancel (value changed mid-tween, or StrictMode re-run) keep the
    // intermediate value so the next tween continues smoothly from it.
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);

  return (
    <span key={flash} className={flash ? 'num-flash' : undefined}>
      {format(shown)}
    </span>
  );
}
