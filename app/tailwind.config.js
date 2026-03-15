/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          0: "#08080d",
          1: "#0e0e16",
          2: "#14141f",
          3: "#1a1a28",
          4: "#222233",
        },
        p1: {
          DEFAULT: "#e84040",
          light: "#ff6b6b",
          dark: "#a82020",
        },
        p2: {
          DEFAULT: "#4088e8",
          light: "#6bb0ff",
          dark: "#2060a8",
        },
        accent: {
          gold: "#f0c040",
          green: "#40c878",
          purple: "#e8c840",
        },
        text: {
          primary: "#e8e8f0",
          secondary: "#8888aa",
          muted: "#555570",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "system-ui",
          "-apple-system",
          "sans-serif",
        ],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
    },
  },
  plugins: [],
};
