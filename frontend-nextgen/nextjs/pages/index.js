// pages/index.js

import Head from 'next/head';
import Link from 'next/link';
import Script from 'next/script';



// Define your airports with their codes and names
const airportNames = {
    cph: '🇩🇰 Copenhagen Airport',
    osl: '🇳🇴 Oslo Airport',
    arn: '🇸🇪 Stockholm Airport',
    dus: '🇩🇪 Düsseldorf Airport',
    fra: '🇩🇪 Frankfurt Airport',
    muc: '🇩🇪 Munich Airport',
    lhr: '🇬🇧 London Heathrow Airport',
    ams: '🇳🇱 Amsterdam Airport',
    dub: '🇮🇪 Dublin Airport',
    ist: '🇹🇷 Istanbul Airport',
  };

export default function Home() {
  return (
    <>
      {/* Head for SEO optimization */}
      <Head>
        <title>Waitport - Real-time & Predicted Airport Security Queues</title>
        <meta
          name="description"
          content="Check live and predicted security queue wait times at major European airports. Plan your trip effectively with Waitport's real-time data and future estimates."
        />
        <meta property="og:title" content="Waitport - Real-time & Predicted Airport Security Queues" />
        <meta
          property="og:description"
          content="Check live and predicted security queue wait times at major European airports. Plan your trip effectively with Waitport's real-time data and future estimates."
        />
        <meta property="og:url" content="https://waitport.com" />
        <meta property="og:type" content="website" />
        <link rel="canonical" href="https://waitport.com" />
        {/* Structured Data for SEO */}
        <link rel="icon" href="/favicon.ico" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "WebSite",
              "name": "Waitport",
              "url": "https://waitport.com",
              "description": "Check live and predicted security queue wait times at major European airports. Plan your trip effectively with Waitport's real-time data and future estimates.",
              "potentialAction": {
                "@type": "SearchAction",
                "target": "https://waitport.com/search?query={search_term_string}",
                "query-input": "required name=search_term_string",
              },
            }),
          }}
        />
      </Head>
        <Script
          src="https://umami.waitport.com/script.js"
          data-website-id="25e1973f-f0c8-489c-bb41-6726ad81ca4d"
          strategy="afterInteractive"
        />

      <div className="bg-gray-50 p-5">
        <h1 className="text-center">Waitport 🛫</h1>
        <h4 className="text-center mb-5">Real-Time & Predicted Airport Security Queues</h4>

        <div className="max-w-5xl mx-auto">
          {/* About Section */}
          <div className="flex justify-center">
            <div className="w-full">
              <h2 className="mb-3">About Waitport</h2>
              <p className="text-lg">
                Welcome to <strong>Waitport</strong>! Here, you can track security waiting times across major European airports in real-time.
                We also provide conservative queue <strong>predictions</strong> for future dates and times. This helps you plan your trip more effectively and avoid unexpected delays.
                <br /><br />
                Select an airport below to see the current and predicted security queue times. <span role="img" aria-label="globe">🌏</span>
                <br /><br />
                Safe travels!
              </p>
              <hr />
            </div>
          </div>

          {/* Airport Selection Section */}
          <div className="flex justify-center">
            <div className="w-full">
              <h2 className="mb-3">Select Airport</h2>
              <ul className="divide-y divide-gray-200 border rounded">
                {Object.entries(airportNames).map(([code, name]) => (
                  <li key={code} className="p-3 hover:bg-gray-100">
                    <Link href={`/airports/${code}`} className="text-decoration-none">
                      {name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Optional: Additional Content or Features */}
          {/* You can add more sections here as needed */}

          {/* Removed Bootstrap divider demo element */}

          {/* Footer Section */}
          <div className="max-w-5xl mx-auto">
            <footer className="py-3 my-4">
              <ul className="flex justify-center border-b pb-3 mb-3">
                <li className="nav-item">
                  <a
                    href="https://simonottosen.dk/"
                    className="mx-2 text-gray-600 hover:text-gray-800"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Other projects
                  </a>
                </li>
                <li className="nav-item">
                  <a
                    href="https://waitport.com/api/v1/all?order=id.desc&limit=100"
                    className="mx-2 text-gray-600 hover:text-gray-800"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    API
                  </a>
                </li>
                <li className="nav-item">
                  <a
                    href="https://github.com/simonottosen/cph-security"
                    className="mx-2 text-gray-600 hover:text-gray-800"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    GitHub
                  </a>
                </li>
              </ul>
              <p className="text-center text-muted">
                Made with <span role="img" aria-label="heart">❤️</span> by Simon Ottosen
              </p>
            </footer>
          </div>
        </div>
      </div>
    </>
  );
}
