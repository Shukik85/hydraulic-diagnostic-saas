/**
 * Cypress component testing support
 */

import { mount } from 'cypress/vue';
import './commands';

// Augment the Cypress namespace to include type definitions for custom commands
declare global {
  namespace Cypress {
    interface Chainable {
      mount: typeof mount;
    }
  }
}

Cypress.Commands.add('mount', mount);
