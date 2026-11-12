import { useState } from 'react';
import { DqnView } from './components/dqn/DqnView';
import { TabularView } from './components/TabularView';
import { Algorithm, AlgorithmSelector } from './components/TopBar';
import { useDqnSimulation } from './useDqn';
import { useSimulation } from './useSimulation';

/**
 * Both learning modes stay mounted in state at all times, so switching
 * algorithms never destroys training results. Each has its own reset.
 */
export default function App() {
  const tabular = useSimulation();
  const dqn = useDqnSimulation();
  const [algo, setAlgo] = useState<Algorithm>('q');

  const switchTo = (a: Algorithm) => {
    tabular.pause();
    dqn.pause();
    setAlgo(a);
  };
  const selector = <AlgorithmSelector value={algo} onChange={switchTo} />;

  return algo === 'q' ? <TabularView ctl={tabular} modeSelector={selector} /> : <DqnView ctl={dqn} tabular={tabular.state} modeSelector={selector} />;
}
