export default function Table({ history }) {
  return (
    <div>
      <h3>實驗紀錄</h3>
      {history.map((item, i) => (
        <div key={i}>
          Acc: {item.metrics.accuracy} | Sharpe: {item.metrics.sharpe}
        </div>
      ))}
    </div>
  );
}
