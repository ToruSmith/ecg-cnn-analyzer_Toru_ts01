import EquityChart from "./EquityChart";
import LossChart from "./LossChart";

export default function ChartPanel({ data }) {
  if (!data) return null;

  return (
    <div>
      <h3>📈 Equity Curve</h3>
      <EquityChart equity={data.equity_curve} />

      <h3>📉 Training Loss</h3>
      <LossChart history={data.history} />
    </div>
  );
}
