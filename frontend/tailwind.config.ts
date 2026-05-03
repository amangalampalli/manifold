import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "Geist", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["Geist Mono", "SFMono-Regular", "ui-monospace", "monospace"]
      },
      colors: {
        carbon: {
          900: "#090b0f",
          950: "#040507"
        },
        cyanOps: "#20d7ff",
        amberOps: "#ff9d2e",
        perturbOps: "#ff4d5e",
        tensionOps: "#57f287",
        controlOps: "#7c8cff"
      },
      boxShadow: {
        cyan: "0 0 34px rgba(32, 215, 255, 0.32)",
        amber: "0 0 34px rgba(255, 157, 46, 0.28)"
      }
    }
  },
  plugins: []
} satisfies Config;
