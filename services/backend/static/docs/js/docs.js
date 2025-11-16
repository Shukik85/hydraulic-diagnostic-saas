/**
 * Documentation System JavaScript
 * Handles interactive features for the docs system
 */

(function() {
  'use strict';

  // ============================================================
  // CODE COPY BUTTONS
  // ============================================================

  function addCopyButtons() {
    const codeBlocks = document.querySelectorAll('.markdown-content pre');
    
    codeBlocks.forEach((block) => {
      // Skip if button already exists
      if (block.querySelector('.copy-button')) return;

      const button = document.createElement('button');
      button.className = 'copy-button';
      button.innerHTML = '📋 Copy';
      button.setAttribute('aria-label', 'Copy code to clipboard');

      button.addEventListener('click', async () => {
        const code = block.querySelector('code');
        const text = code ? code.textContent : block.textContent;

        try {
          await navigator.clipboard.writeText(text);
          button.innerHTML = '✓ Copied!';
          button.classList.add('copied');
          
          setTimeout(() => {
            button.innerHTML = '📋 Copy';
            button.classList.remove('copied');
          }, 2000);
        } catch (err) {
          console.error('Failed to copy:', err);
          button.innerHTML = '❌ Failed';
          setTimeout(() => {
            button.innerHTML = '📋 Copy';
          }, 2000);
        }
      });

      // Wrap pre block for positioning
      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';
      block.parentNode.insertBefore(wrapper, block);
      wrapper.appendChild(block);
      wrapper.appendChild(button);
    });
  }

  // ============================================================
  // TABLE OF CONTENTS
  // ============================================================

  function generateTableOfContents() {
    const article = document.querySelector('.markdown-content');
    if (!article) return;

    const headings = article.querySelectorAll('h2, h3');
    if (headings.length < 3) return; // Only show TOC if there are enough headings

    const toc = document.createElement('nav');
    toc.className = 'docs-toc';
    toc.innerHTML = '<h3 class="docs-toc__title">Table of Contents</h3>';

    const list = document.createElement('ul');
    list.className = 'docs-toc__list';

    headings.forEach((heading, index) => {
      // Add ID if not present
      if (!heading.id) {
        heading.id = `heading-${index}`;
      }

      const item = document.createElement('li');
      item.className = `docs-toc__item docs-toc__item--${heading.tagName.toLowerCase()}`;

      const link = document.createElement('a');
      link.href = `#${heading.id}`;
      link.textContent = heading.textContent;
      link.className = 'docs-toc__link';

      // Smooth scroll
      link.addEventListener('click', (e) => {
        e.preventDefault();
        heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.pushState(null, '', `#${heading.id}`);
      });

      item.appendChild(link);
      list.appendChild(item);
    });

    toc.appendChild(list);

    // Insert TOC before article content
    const articleHeader = document.querySelector('.docs-article__content');
    if (articleHeader) {
      articleHeader.insertBefore(toc, articleHeader.firstChild);
    }
  }

  // ============================================================
  // SEARCH HIGHLIGHTING
  // ============================================================

  function highlightSearchResults() {
    const urlParams = new URLSearchParams(window.location.search);
    const query = urlParams.get('q');
    
    if (!query || query.length < 3) return;

    const content = document.querySelectorAll('.doc-list-item__title, .doc-list-item__summary');
    const terms = query.toLowerCase().split(/\s+/);

    content.forEach((element) => {
      const text = element.textContent;
      if (!text) return;
      // Prepare regex for terms >= 3 chars
      const validTerms = terms.filter(term => term.length >= 3);
      if (validTerms.length === 0) return;
      // Build RegExp for all valid terms, escape regex special characters
      const escapeRegex = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const combinedRegex = new RegExp("(" + validTerms.map(escapeRegex).join("|") + ")", "gi");
      // Split and highlight parts
      const frag = document.createDocumentFragment();
      let lastIndex = 0;
      let m;
      while ((m = combinedRegex.exec(text)) !== null) {
        if (m.index > lastIndex) {
          frag.appendChild(document.createTextNode(text.substring(lastIndex, m.index)));
        }
        const mark = document.createElement('mark');
        mark.textContent = m[0];
        frag.appendChild(mark);
        lastIndex = combinedRegex.lastIndex;
      }
      if (lastIndex < text.length) {
        frag.appendChild(document.createTextNode(text.substring(lastIndex)));
      }
      // Only update if there were any matches
      if (lastIndex > 0) {
        element.innerHTML = '';
        element.appendChild(frag);
      }
    });
  }

  // ============================================================
  // READING PROGRESS
  // ============================================================

  function initReadingProgress() {
    const article = document.querySelector('.docs-article');
    if (!article) return;

    const progressBar = document.createElement('div');
    progressBar.className = 'reading-progress';
    progressBar.innerHTML = '<div class="reading-progress__bar"></div>';
    document.body.appendChild(progressBar);

    const bar = progressBar.querySelector('.reading-progress__bar');

    function updateProgress() {
      const scrollTop = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const progress = (scrollTop / docHeight) * 100;
      bar.style.width = `${Math.min(progress, 100)}%`;
    }

    window.addEventListener('scroll', updateProgress, { passive: true });
    updateProgress();
  }

  // ============================================================
  // KEYBOARD SHORTCUTS
  // ============================================================

  function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Cmd/Ctrl + K: Focus search
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('.docs-search-mini__input, .docs-search-form__input');
        if (searchInput) {
          searchInput.focus();
          searchInput.select();
        }
      }
    });
  }

  // ============================================================
  // MOBILE MENU TOGGLE
  // ============================================================

  function initMobileMenu() {
    if (window.innerWidth > 1024) return;

    const sidebar = document.querySelector('.docs-sidebar');
    if (!sidebar) return;

    const toggle = document.createElement('button');
    toggle.className = 'docs-menu-toggle';
    toggle.innerHTML = '☰ Menu';
    toggle.setAttribute('aria-label', 'Toggle navigation menu');

    toggle.addEventListener('click', () => {
      sidebar.classList.toggle('docs-sidebar--open');
      toggle.classList.toggle('active');
    });

    document.querySelector('.docs-main')?.prepend(toggle);
  }

  // ============================================================
  // EXTERNAL LINK ICONS
  // ============================================================

  function markExternalLinks() {
    const links = document.querySelectorAll('.markdown-content a[href^="http"]');
    
    links.forEach((link) => {
      if (!link.hostname.includes(window.location.hostname)) {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
        link.classList.add('external-link');
        
        if (!link.querySelector('.external-icon')) {
          const icon = document.createElement('span');
          icon.className = 'external-icon';
          icon.innerHTML = ' ↗';
          link.appendChild(icon);
        }
      }
    });
  }

  // ============================================================
  // SCROLL TO TOP
  // ============================================================

  function initScrollToTop() {
    const button = document.createElement('button');
    button.className = 'scroll-to-top';
    button.innerHTML = '↑';
    button.setAttribute('aria-label', 'Scroll to top');
    document.body.appendChild(button);

    function toggleButton() {
      if (window.scrollY > 400) {
        button.classList.add('visible');
      } else {
        button.classList.remove('visible');
      }
    }

    button.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    window.addEventListener('scroll', toggleButton, { passive: true });
    toggleButton();
  }

  // ============================================================
  // INITIALIZATION
  // ============================================================

  function init() {
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
      return;
    }

    // Initialize features
    addCopyButtons();
    generateTableOfContents();
    highlightSearchResults();
    initReadingProgress();
    initKeyboardShortcuts();
    initMobileMenu();
    markExternalLinks();
    initScrollToTop();

    // Add CSS for dynamic elements
    injectStyles();
  }

  function injectStyles() {
    const styles = `
      .code-block-wrapper {
        position: relative;
        margin-bottom: var(--space-16, 16px);
      }

      .copy-button {
        position: absolute;
        top: 8px;
        right: 8px;
        padding: 4px 12px;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 4px;
        font-size: 0.75rem;
        cursor: pointer;
        transition: all 0.2s ease;
      }

      .copy-button:hover {
        background: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      }

      .copy-button.copied {
        background: #10B981;
        color: white;
        border-color: #10B981;
      }

      .docs-toc {
        margin-bottom: var(--space-32, 32px);
        padding: var(--space-20, 20px);
        background: rgba(33, 128, 141, 0.05);
        border-radius: var(--radius-base, 8px);
        border-left: 3px solid var(--color-primary, #21808D);
      }

      .docs-toc__title {
        margin: 0 0 var(--space-12, 12px) 0;
        font-size: 1rem;
        font-weight: 600;
      }

      .docs-toc__list {
        list-style: none;
        padding: 0;
        margin: 0;
      }

      .docs-toc__item {
        margin-bottom: var(--space-8, 8px);
      }

      .docs-toc__item--h3 {
        padding-left: var(--space-16, 16px);
      }

      .docs-toc__link {
        color: var(--color-text, #134252);
        text-decoration: none;
        font-size: 0.875rem;
        transition: color 0.2s ease;
      }

      .docs-toc__link:hover {
        color: var(--color-primary, #21808D);
      }

      .reading-progress {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: rgba(33, 128, 141, 0.1);
        z-index: 1000;
      }

      .reading-progress__bar {
        height: 100%;
        background: var(--color-primary, #21808D);
        width: 0;
        transition: width 0.1s ease;
      }

      .external-link .external-icon {
        opacity: 0.6;
        font-size: 0.875em;
      }

      .scroll-to-top {
        position: fixed;
        bottom: 24px;
        right: 24px;
        width: 48px;
        height: 48px;
        background: var(--color-primary, #21808D);
        color: white;
        border: none;
        border-radius: 50%;
        font-size: 1.5rem;
        cursor: pointer;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z-index: 999;
      }

      .scroll-to-top.visible {
        opacity: 1;
        visibility: visible;
      }

      .scroll-to-top:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
      }

      .docs-menu-toggle {
        display: none;
        padding: var(--space-8, 8px) var(--space-16, 16px);
        background: var(--color-surface, #FFFFFD);
        border: 1px solid var(--color-border, rgba(94,82,64,0.2));
        border-radius: var(--radius-base, 8px);
        font-size: 0.875rem;
        cursor: pointer;
        margin-bottom: var(--space-16, 16px);
      }

      @media (max-width: 1024px) {
        .docs-menu-toggle {
          display: block;
        }

        .docs-sidebar {
          display: none;
        }

        .docs-sidebar--open {
          display: block;
        }
      }

      mark {
        background: rgba(255, 237, 0, 0.4);
        padding: 0 2px;
        border-radius: 2px;
      }
    `;

    const styleEl = document.createElement('style');
    styleEl.textContent = styles;
    document.head.appendChild(styleEl);
  }

  // Start initialization
  init();
})();
