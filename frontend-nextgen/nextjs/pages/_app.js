// pages/_app.js
import 'bootstrap/dist/css/bootstrap.min.css';
import 'react-datetime/css/react-datetime.css';
import 'moment/locale/da'; // Import moment locale globally

function Waitport({ Component, pageProps }) {
  return <Component {...pageProps} />;
}

export default Waitport;