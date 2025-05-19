// pages/_app.js
import "tailwindcss";
import 'react-datetime/css/react-datetime.css';
import 'moment/locale/da'; // Import moment locale globally

function Waitport({ Component, pageProps }) {
  return <Component {...pageProps} />;
}

export default Waitport;