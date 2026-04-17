export default function ChartPanel({ data }) {
  if (!data) return null;

  return (
    <div>
      <h3>Loss</h3>
      <pre>{JSON.stringify(data.history, null, 2)}</pre>

      <h3>Equity Curve</h3>
      <pre>{JSON.stringify(data.equity_curve)}</pre>
    </div>
  );
}
