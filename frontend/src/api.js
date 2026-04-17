const BASE_URL = "https://你的-render-url.onrender.com";

export async function runExperiment(params) {
  const res = await fetch(`${BASE_URL}/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(params)
  });

  return res.json();
}
