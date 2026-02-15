/**
 * GRAVIS Document Viewer
 *
 * Reusable module that opens a rendered markdown document in the
 * right panel area, replacing the toggle bar + chart with a tabbed
 * document view at the same hierarchy level as .right-panel.
 *
 * Usage from any page:
 *   GravisDoc.open('hubble-tension')        // opens docs/hubble-tension.md
 *   GravisDoc.open('hubble-tension', {
 *       panel: '#my-panel',                 // custom panel selector
 *       onClose: function() { ... }         // callback when closed
 *   })
 *   GravisDoc.close()                       // close and restore panel
 *
 * Requirements:
 *   - marked.js loaded (CDN in base.html)
 *   - gravis.css doc-viewer styles loaded
 *   - A .right-panel element (or custom panel)
 */

var GravisDoc = (function() {
    'use strict';

    var FONT_SIZES = [0.82, 0.88, 0.92, 1.0, 1.08, 1.16, 1.24];
    var fontIndex = 2; // default 0.92em

    var state = {
        isOpen: false,
        viewer: null,
        rightPanel: null,
        onClose: null,
        currentDoc: null
    };

    function applyFontSize(viewer) {
        var body = viewer.querySelector('.doc-body');
        if (body) {
            body.style.fontSize = FONT_SIZES[fontIndex] + 'em';
        }
    }

    function getOrCreateViewer(panelSel) {
        var panel = panelSel
            ? document.querySelector(panelSel)
            : document.querySelector('.right-panel');

        if (!panel) {
            console.error('GravisDoc: no .right-panel found');
            return null;
        }

        state.rightPanel = panel;

        // The viewer is a sibling of .right-panel inside .analysis-body
        var parent = panel.parentElement;
        var viewer = parent.querySelector('#doc-viewer');
        if (!viewer) {
            viewer = document.createElement('div');
            viewer.id = 'doc-viewer';
            viewer.className = 'doc-viewer';
            viewer.style.display = 'none';

            viewer.innerHTML =
                '<div class="doc-tab-bar">' +
                    '<div class="doc-tab active">' +
                        '<span class="doc-tab-icon">&#9776;</span>' +
                        '<span class="doc-tab-name"></span>' +
                        '<button class="doc-tab-close" title="Close document">&times;</button>' +
                    '</div>' +
                '</div>' +
                '<div class="doc-content">' +
                    '<div class="doc-font-toolbar">' +
                        '<button class="doc-font-btn" id="doc-font-down" title="Decrease font size">A&#8722;</button>' +
                        '<button class="doc-font-btn" id="doc-font-up" title="Increase font size">A+</button>' +
                    '</div>' +
                    '<div class="doc-body"></div>' +
                '</div>';

            parent.appendChild(viewer);

            viewer.querySelector('.doc-tab-close').addEventListener('click', function() {
                close();
            });

            viewer.querySelector('#doc-font-down').addEventListener('click', function() {
                if (fontIndex > 0) {
                    fontIndex--;
                    applyFontSize(viewer);
                }
            });

            viewer.querySelector('#doc-font-up').addEventListener('click', function() {
                if (fontIndex < FONT_SIZES.length - 1) {
                    fontIndex++;
                    applyFontSize(viewer);
                }
            });
        }

        state.viewer = viewer;
        return viewer;
    }

    function open(docName, options) {
        options = options || {};

        var viewer = getOrCreateViewer(options.panel || null);
        if (!viewer) return;

        state.onClose = options.onClose || null;
        state.currentDoc = docName;

        var tabName = viewer.querySelector('.doc-tab-name');
        var body = viewer.querySelector('.doc-body');

        var displayName = docName.replace(/\.md$/, '') + '.md';
        tabName.textContent = displayName;
        body.innerHTML = '<div class="doc-loading">Loading...</div>';

        // Hide the right panel, show the viewer in its place
        if (state.rightPanel) {
            state.rightPanel.style.display = 'none';
        }
        viewer.style.display = 'flex';
        state.isOpen = true;

        // Fetch and render markdown
        var url = '/api/doc/' + encodeURIComponent(docName);
        fetch(url)
            .then(function(resp) {
                if (!resp.ok) throw new Error('Document not found: ' + docName);
                return resp.text();
            })
            .then(function(md) {
                if (typeof marked !== 'undefined' && marked.parse) {
                    body.innerHTML = marked.parse(md);
                } else {
                    body.innerHTML = '<pre>' + md + '</pre>';
                }
            })
            .catch(function(err) {
                body.innerHTML =
                    '<div class="doc-error">' +
                        '<strong>Error</strong><br>' + err.message +
                    '</div>';
            });
    }

    function close() {
        if (!state.isOpen || !state.viewer) return;

        state.viewer.style.display = 'none';

        // Restore the right panel
        if (state.rightPanel) {
            state.rightPanel.style.display = '';
        }

        state.isOpen = false;
        state.currentDoc = null;

        if (typeof state.onClose === 'function') {
            state.onClose();
        }
    }

    function isOpen() {
        return state.isOpen;
    }

    function currentDoc() {
        return state.currentDoc;
    }

    return {
        open: open,
        close: close,
        isOpen: isOpen,
        currentDoc: currentDoc
    };
})();
