/**
 * Cypress E2E support file
 */

// Import commands
import './commands';

// Custom error handler
Cypress.on('uncaught:exception', (err) => {
  // Prevent Cypress from failing tests on uncaught exceptions
  // You can add specific error handling here
  console.error('Uncaught exception:', err);
  return false;
});

// Before each test
beforeEach(() => {
  // Clear localStorage
  cy.clearLocalStorage();

  // Clear cookies
  cy.clearCookies();
});
