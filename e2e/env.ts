import 'dotenv/config';

const target = (process.env.TARGET ?? 'local').toLowerCase();
if (target !== 'local' && target !== 'prod') {
  throw new Error(`TARGET must be "local" or "prod", got "${target}"`);
}

const pick = (key: string): string => {
  const v = process.env[`${target.toUpperCase()}_${key}`];
  if (!v) throw new Error(`Missing ${target.toUpperCase()}_${key} in .env`);
  return v.replace(/\/+$/, ''); // strip trailing slash — CORS origins must match exactly
};

export const ENV = {
  target,
  frontendUrl: pick('FRONTEND_URL'),
  backendUrl: pick('BACKEND_URL'),
  email: process.env.TEST_EMAIL ?? '',
  password: process.env.TEST_PASSWORD ?? '',
  question:
    process.env.TEST_QUESTION ??
    'Is an M-Pesa statement admissible as evidence in the Small Claims Court?',
};

export const AUTH_FILE = `.auth/advocate-${target}.json`;

// ChatView.doRequest() fires these two in order on every submit (both plain JSON, not streamed).
export const RETRIEVE_API = /\/api\/qretrieve(\?|$)/;
export const CHAT_API = /\/api\/generate-legal-answer(\?|$)/; // response.answer holds the text

// Visible strings from the UI (ChatView.tsx / auth/sign-in/page.tsx)
export const UI = {
  loading: 'Drafting grounds from authorities…',
  assistantHeader: 'Legal Analysis',
  chatError: 'Request failed',
  signInErrors: {
    badCreds: 'Invalid email or password.',
    noUser: 'No account found with this email.',
    rateLimited: 'Too many attempts — please try again later.',
    generic: 'An error occurred. Please try again.', // unauthorized-domain lands here
  },
};
