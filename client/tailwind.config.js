/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark': '#17191F',
        'white-1': '#FAFAFA',
        'white-2': '#FDFDFD',
        'white-3': '#F0F0F0',
        'black-1': '#1F1F1F',
      },
      fontFamily: {
        'spaceGrotesk': ['Space Grotesk', 'sans-serif'],
      }
    },
  },
  plugins: [],
}

