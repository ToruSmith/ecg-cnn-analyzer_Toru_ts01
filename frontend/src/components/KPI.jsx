export default function KPI({ data }) {
  if (!data) return null;

  return (
    <div>
      <h2>Accuracy: {data.metrics.accuracy}</h2>
      <h2>Sharpe: {data.metrics.sharpe}</h2>
    </div>
  );
}
