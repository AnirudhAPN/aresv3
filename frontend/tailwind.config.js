/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0b0e1a",
        surface: "#12172a",
        surface2: "#1a2036",
        border: "#242c47",
        primary: {
          DEFAULT: "#7c5cff",
          50: "#f3f0ff",
          400: "#9a80ff",
          500: "#7c5cff",
          600: "#6142e6",
          700: "#4b2ea8",
        },
        accent: {
          DEFAULT: "#3ec6ff",
          500: "#3ec6ff",
        },
        muted: "#8892b0",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      boxShadow: {
        glow: "0 0 24px 0 rgba(124, 92, 255, 0.25)",
      },
    },
  },
  plugins: [],
};
