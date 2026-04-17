import ReactECharts from "echarts-for-react";

export default function EquityChart({ equity }) {
  if (!equity) return null;

  const option = {
    xAxis: {
      type: "category",
      data: equity.map((_, i) => i)
    },
    yAxis: {
      type: "value"
    },
    series: [
      {
        data: equity,
        type: "line",
        smooth: true
      }
    ]
  };

  return <ReactECharts option={option} style={{ height: 300 }} />;
}
