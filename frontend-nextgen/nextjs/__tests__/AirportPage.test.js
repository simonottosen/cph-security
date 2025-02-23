// __tests__/AirportPage.test.js
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import AirportPage, { getStaticProps, getStaticPaths } from '../pages/airports/[code]';
import axios from 'axios';
import '@testing-library/jest-dom';

// Mock axios
jest.mock('axios');

// Mock next/dynamic to avoid issues with dynamic imports in tests.
// This replaces the DateTime component with a dummy component.
jest.mock('next/dynamic', () => (importFn, options) => {
  return function DummyDynamicComponent(props) {
    return <div data-testid="datetime-component">Dynamic Component</div>;
  };
});

describe('AirportPage Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders header and displays queue information', async () => {
    // Mock responses for axios requests based on URL patterns.
    const queueResponse = { data: [{ queue: 10 }] };
    const averageResponse = { data: [{ queue: 8 }, { queue: 12 }] };
    const predictedResponse = { data: { predicted_queue_length_minutes: 15 } };

    axios.get.mockImplementation((url) => {
      if (url.includes('limit=1')) {
        return Promise.resolve(queueResponse);
      }
      if (url.includes('limit=24')) {
        return Promise.resolve(averageResponse);
      }
      if (url.includes('/predict')) {
        return Promise.resolve(predictedResponse);
      }
      return Promise.reject(new Error('Not Found'));
    });

    // Get static props for airport code 'cph'
    const { props } = await getStaticProps({ params: { code: 'cph' } });
    render(<AirportPage {...props} />);

    // Verify header renders
    expect(screen.getByText(/Waitport 🛫/)).toBeInTheDocument();

    // Wait for the "Current Security Queue" heading to appear.
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: /Current Security Queue/ })).toBeInTheDocument()
    );

    // Get the heading element and then the paragraph following it.
    const currentHeading = screen.getByRole('heading', { name: /Current Security Queue/ });
    // Assuming the paragraph is a sibling within the same parent.
    const currentQueueParagraph = currentHeading.parentElement.querySelector('p');

    expect(currentQueueParagraph).toBeInTheDocument();
    // Check that the paragraph text contains "is currently" followed by "10 minutes"
    expect(currentQueueParagraph.textContent).toMatch(/is currently\s+10 minutes/);

    // Wait for predicted queue info to load and verify its value.
    await waitFor(() =>
      expect(screen.getByText(/Predicted Security Queue/)).toBeInTheDocument()
    );
    expect(screen.getByText(/15 minutes/)).toBeInTheDocument();
  });
});

describe('getStaticPaths', () => {
  test('returns valid paths and fallback is false', async () => {
    const pathsResult = await getStaticPaths();
    expect(pathsResult.paths).toEqual(
      expect.arrayContaining([
        { params: { code: 'cph' } },
        { params: { code: 'osl' } },
        { params: { code: 'arn' } },
        { params: { code: 'dus' } },
        { params: { code: 'fra' } },
        { params: { code: 'muc' } },
        { params: { code: 'lhr' } },
        { params: { code: 'ams' } },
        { params: { code: 'dub' } },
        { params: { code: 'ist' } },
      ])
    );
    expect(pathsResult.fallback).toBe(false);
  });
});

describe('getStaticProps', () => {
  test('returns correct props for a known airport code', async () => {
    const result = await getStaticProps({ params: { code: 'cph' } });
    expect(result.props).toHaveProperty('code', 'cph');
    expect(result.props).toHaveProperty('airportName', '🇩🇰 Copenhagen Airport');
  });

  test('returns "Unknown Airport" for an unknown airport code', async () => {
    const result = await getStaticProps({ params: { code: 'unknown' } });
    expect(result.props).toHaveProperty('code', 'unknown');
    expect(result.props).toHaveProperty('airportName', 'Unknown Airport');
  });
});