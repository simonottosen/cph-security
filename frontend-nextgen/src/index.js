import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import Pant from './Pant'; // Assuming Pant is in the same directory
import reportWebVitals from './reportWebVitals';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <Router>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/pant" element={<Pant />} />
      </Routes>
    </Router>
  </React.StrictMode>
);

reportWebVitals();
