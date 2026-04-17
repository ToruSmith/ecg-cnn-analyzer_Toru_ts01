import ReactECharts from "echarts-for-react";

export default function LossChart({ history }) {
  if (!history) return null;

  const option = {
    legend: { data: ["loss", "val_loss"] },
    xAxis: { type: "category", data: history.loss.map((_, i) => i) },
    yAxis: { type: "value" },
    series: [
      { name: "loss", data: history.loss, type: "line" },
      { name: "val_loss", data: history.val_loss, type: "line" }
    ]
  };

  return <ReactECharts option={option} style={{ height: 300 }} />;
}
