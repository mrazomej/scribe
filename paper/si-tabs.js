/*
 * si-tabs.js
 *
 * Converts the flat Supplementary Information document into a tabbed
 * interface with a synchronised sidebar Table of Contents (TOC).
 *
 * Quarto renders every included .qmd file into a single monolithic HTML
 * page.  This script, loaded at the end of <body>, restructures the page
 * at runtime so that each thematic block (Core Model, Inference, etc.)
 * lives in its own Bootstrap tab panel.
 *
 * Overview of what the script does:
 *
 *   1. Finds the .si-tab-container div and its .si-tab children.
 *   2. Injects a CSS rule to hide Quarto's auto-generated section
 *      numbers (which are meaningless after the heading-level shift).
 *   3. Creates a sticky Bootstrap nav-tabs bar from each panel's
 *      data-tab-title attribute.
 *   4. Wraps every panel in a .tab-content container and marks the
 *      first panel as the initially active one.
 *   5. Builds a lookup table mapping every section ID in the document
 *      to the tab index that contains it.
 *   6. Replaces Quarto's sparse sidebar TOC (which cannot see headings
 *      inside Pandoc Divs) with a custom TOC built from the actual
 *      headings in each panel.
 *   7. Filters the TOC so only entries for the active tab are visible.
 *   8. Listens for tab switches (Bootstrap "shown.bs.tab") and updates
 *      the TOC filter accordingly.
 *   9. Intercepts TOC link clicks: if the target section lives in a
 *      different tab, activates that tab first, then scrolls.
 *  10. Uses an IntersectionObserver to highlight the currently visible
 *      section in the sidebar TOC as the user scrolls.
 *
 * Companion file: si-tabs.css (loaded in <head> to prevent FOUC).
 */

document.addEventListener("DOMContentLoaded", function () {

  // ── Locate the tab container and its panels ─────────────────────────
  // The .si-tab-container is a Pandoc fenced div written in
  // supplementary.qmd.  Each .si-tab child corresponds to one thematic
  // block (e.g. "I. Core Model").
  var container = document.querySelector(".si-tab-container");
  if (!container) return;

  var panels = Array.from(container.querySelectorAll(":scope > .si-tab"));
  if (panels.length === 0) return;

  // ── Hide auto-generated section numbers ─────────────────────────────
  // Quarto's crossref system still numbers sections internally (needed
  // for @sec-* references to resolve), but the rendered "Section 0.0.X"
  // labels are meaningless in the tabbed layout.  This rule hides the
  // <span class="header-section-number"> elements that Quarto injects
  // into heading text.
  var style = document.createElement("style");
  style.textContent =
    ".si-tab-container .header-section-number { display: none; }";
  document.head.appendChild(style);

  // ── Build the sticky tab navigation bar ─────────────────────────────
  // Creates a <ul class="nav nav-tabs"> with one button per panel.
  // The bar is sticky so it stays visible as the user scrolls through
  // long sections.
  var nav = document.createElement("ul");
  nav.className = "nav nav-tabs mb-4";
  nav.setAttribute("role", "tablist");
  nav.style.position = "sticky";
  nav.style.top = "0";
  nav.style.zIndex = "10";
  nav.style.backgroundColor = "var(--bs-body-bg, #fff)";
  nav.style.paddingTop = "0.5rem";

  panels.forEach(function (panel, i) {
    // Read the human-friendly title from the data attribute set in the
    // .qmd source (e.g. data-tab-title="I. Core Model")
    var title = panel.getAttribute("data-tab-title") || "Tab " + (i + 1);

    // Assign each panel an ID and Bootstrap tab-pane classes.
    // The first panel starts as the active/visible one.
    var id = "si-tab-" + i;
    panel.id = id;
    panel.setAttribute("role", "tabpanel");
    panel.classList.add("tab-pane", "fade");
    if (i === 0) panel.classList.add("show", "active");

    // Build the <li><button> for this tab
    var li = document.createElement("li");
    li.className = "nav-item";
    li.setAttribute("role", "presentation");

    var btn = document.createElement("button");
    btn.className = "nav-link" + (i === 0 ? " active" : "");
    btn.setAttribute("data-bs-toggle", "tab");
    btn.setAttribute("data-bs-target", "#" + id);
    btn.setAttribute("type", "button");
    btn.setAttribute("role", "tab");
    btn.textContent = title;
    li.appendChild(btn);
    nav.appendChild(li);
  });

  // ── Wrap panels in a Bootstrap .tab-content container ───────────────
  // Move all panels into a single wrapper div so Bootstrap's tab logic
  // can show/hide them correctly.
  var tabContent = document.createElement("div");
  tabContent.className = "tab-content";
  panels.forEach(function (panel) {
    tabContent.appendChild(panel);
  });

  // Replace the original container contents with the tab bar + panels
  container.innerHTML = "";
  container.appendChild(nav);
  container.appendChild(tabContent);

  // ── Build section-ID → tab-index lookup ─────────────────────────────
  // Every element with an id inside each panel is mapped to that panel's
  // index.  Used later to figure out which tab a TOC link or cross-
  // reference target belongs to.
  var sectionToTab = {};
  panels.forEach(function (panel, i) {
    panel.querySelectorAll("[id]").forEach(function (el) {
      sectionToTab[el.id] = i;
    });
    sectionToTab[panel.id] = i;
  });

  // ── Build the custom sidebar TOC ────────────────────────────────────
  // Quarto's built-in TOC generator cannot see headings inside Pandoc
  // Divs, so the rendered TOC is nearly empty.  We replace it entirely
  // with a TOC built from the actual headings in the DOM.

  var tocNav = document.getElementById("TOC") ||
               document.querySelector("nav.toc-active");
  if (!tocNav) return;

  // Find the shallowest heading level across all panels so we can
  // normalise nesting (e.g. if the shallowest heading is h1, then h1
  // entries become "top-level" TOC items, h2 become children, etc.)
  var minLevel = 9;
  panels.forEach(function (panel) {
    panel.querySelectorAll("h1,h2,h3,h4,h5,h6").forEach(function (h) {
      var lvl = parseInt(h.tagName.substring(1));
      if (lvl < minLevel) minLevel = lvl;
    });
  });

  // Include two levels of headings: the top-level sections plus one
  // level of subsections.  Subsections start collapsed and expand
  // accordion-style when the user clicks their parent section.
  var maxLevel = minLevel + 1;

  // Walk each panel and collect its eligible headings.
  // Skip .unnumbered headings — those are the block organiser titles
  // (e.g. "I. Core Model") which get their own top-level TOC entry.
  // Also skip anything deeper than maxLevel to keep the sidebar compact.
  var panelHeadings = panels.map(function (panel) {
    var headings = [];
    panel.querySelectorAll("h1,h2,h3,h4,h5,h6").forEach(function (h) {
      if (h.classList.contains("unnumbered")) return;
      var lvl = parseInt(h.tagName.substring(1));
      if (lvl > maxLevel) return;
      var id = h.getAttribute("id") ||
               h.closest("section[id]")?.getAttribute("id") || "";
      if (!id) return;
      headings.push({ level: lvl, text: h.textContent, id: id });
    });
    return headings;
  });

  // ── buildTocTree: convert a flat heading list into nested <ul>/<li> ─
  // Uses a stack to track the current nesting depth.  Each heading
  // opens a new <li> at the correct level and prepares a child <ul>
  // for potential sub-headings.  Child <ul>s start hidden and are
  // toggled open/closed via accordion logic (see below).
  // Empty trailing <ul>s are pruned at the end.
  function buildTocTree(headings, tabIdx) {
    var root = document.createElement("ul");
    root.className = "collapse show";
    var stack = [{ ul: root, level: minLevel - 1 }];

    headings.forEach(function (h) {
      // Unwind the stack until we find the correct parent level
      while (stack.length > 1 && stack[stack.length - 1].level >= h.level) {
        stack.pop();
      }

      // Create the <li> with a nav-link <a> matching Quarto's TOC style
      var li = document.createElement("li");
      li.setAttribute("data-si-tab-idx", tabIdx);

      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.className = "nav-link";
      a.setAttribute("data-scroll-target", "#" + h.id);
      a.innerHTML = h.text;
      li.appendChild(a);

      stack[stack.length - 1].ul.appendChild(li);

      // Push a child <ul> for any sub-headings that may follow.
      // Starts hidden — the accordion click handler will reveal it
      // by adding the "si-expanded" class.
      var childUl = document.createElement("ul");
      childUl.className = "si-toc-sub";
      li.appendChild(childUl);
      stack.push({ ul: childUl, level: h.level });
    });

    // Clean up <ul> elements that ended up with no children
    root.querySelectorAll("ul").forEach(function (ul) {
      if (ul.children.length === 0) ul.remove();
    });

    return root;
  }

  // ── Replace Quarto's TOC with the custom one ────────────────────────
  // Preserve the "On this page" title element, clear everything else,
  // then append one top-level <li> per tab (the block header) with its
  // sub-heading tree underneath.

  var titleEl = tocNav.querySelector("#toc-title");
  while (tocNav.firstChild) tocNav.removeChild(tocNav.firstChild);
  if (titleEl) tocNav.appendChild(titleEl);

  var outerUl = document.createElement("ul");

  panels.forEach(function (panel, i) {
    // Use the tab title as the top-level TOC label for this block
    var title = panel.getAttribute("data-tab-title") ||
                panels[i].querySelector("h2")?.textContent ||
                "Tab " + (i + 1);

    var blockLi = document.createElement("li");
    blockLi.setAttribute("data-si-tab-idx", i);

    var blockA = document.createElement("a");
    blockA.href = "#" + panel.id;
    blockA.className = "nav-link" + (i === 0 ? " active" : "");
    blockA.setAttribute("data-scroll-target", "#" + panel.id);
    blockA.textContent = title;
    blockLi.appendChild(blockA);

    // Nest the section headings under the block header
    if (panelHeadings[i].length > 0) {
      var tree = buildTocTree(panelHeadings[i], i);
      blockLi.appendChild(tree);
    }

    outerUl.appendChild(blockLi);
  });

  tocNav.appendChild(outerUl);

  // ── Accordion behaviour for TOC sections ────────────────────────────
  // Clicking a top-level section heading (one that has a nested <ul>)
  // expands its subsections and collapses any other expanded section
  // within the same tab.  This keeps the sidebar compact while still
  // giving access to all subsections.
  tocNav.querySelectorAll("a.nav-link").forEach(function (link) {
    var parentLi = link.closest("li");
    if (!parentLi) return;
    var childUl = parentLi.querySelector(":scope > ul.si-toc-sub");
    if (!childUl) return;

    link.addEventListener("click", function () {
      var isOpen = childUl.classList.contains("si-expanded");

      // Collapse every sibling section's sub-list first (accordion)
      var siblings = parentLi.parentElement.querySelectorAll(
        ":scope > li > ul.si-toc-sub"
      );
      siblings.forEach(function (ul) { ul.classList.remove("si-expanded"); });

      // Toggle: if it was closed, open it; if already open, stay closed
      if (!isOpen) {
        childUl.classList.add("si-expanded");
      }
    });
  });

  // Collect every link in our custom TOC for use by event handlers below
  var allTocLinks = Array.from(
    tocNav.querySelectorAll("a.nav-link[href^='#']")
  );

  // ── filterToc: show/hide TOC entries by active tab ──────────────────
  // Each <li> carries a data-si-tab-idx attribute.  We simply toggle
  // display:none on items that don't belong to the active tab.
  function filterToc(activeIdx) {
    outerUl.querySelectorAll("li[data-si-tab-idx]").forEach(function (li) {
      li.style.display =
        parseInt(li.getAttribute("data-si-tab-idx")) === activeIdx
          ? ""
          : "none";
    });
  }

  // Show only the first tab's TOC entries on initial load
  filterToc(0);

  // ── Tab switch listener ─────────────────────────────────────────────
  // When Bootstrap fires "shown.bs.tab", update the sidebar TOC to
  // reflect the newly active tab and scroll the tab container into view.
  nav.addEventListener("shown.bs.tab", function (e) {
    var target = e.target.getAttribute("data-bs-target");
    var idx = parseInt(target.replace("#si-tab-", ""));
    filterToc(idx);
    container.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  // ── TOC click → activate the correct tab first ──────────────────────
  // If the user clicks a TOC link that belongs to a tab other than the
  // currently active one, we intercept the click, switch tabs via
  // Bootstrap's API, wait briefly for the DOM to update, then scroll
  // to the target section.
  allTocLinks.forEach(function (link) {
    link.addEventListener("click", function (e) {
      var href = link.getAttribute("href").substring(1);
      var tabIdx = sectionToTab[href];
      if (tabIdx === undefined) return;

      // Determine which tab is currently active
      var activeBtn = nav.querySelector(".nav-link.active");
      var activeTarget = activeBtn
        ? activeBtn.getAttribute("data-bs-target")
        : "";
      var activeIdx = parseInt(activeTarget.replace("#si-tab-", "")) || 0;

      // If already on the right tab, let the browser handle the scroll
      if (tabIdx !== activeIdx) {
        e.preventDefault();
        var tabBtn = nav.querySelectorAll(".nav-link")[tabIdx];
        if (tabBtn) {
          var bsTab = new bootstrap.Tab(tabBtn);
          bsTab.show();
          // Small delay for Bootstrap to finish the tab transition
          setTimeout(function () {
            var el = document.getElementById(href);
            if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
          }, 150);
        }
      }
    });
  });

  // ── Scroll-spy: highlight the active TOC entry on scroll ────────────
  // An IntersectionObserver watches every <section> inside the tab
  // panels.  When a section scrolls into the top 20% of the viewport,
  // its corresponding TOC link gets the "active" class.
  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        var id = entry.target.getAttribute("id");
        if (!id) return;
        allTocLinks.forEach(function (l) {
          l.classList.toggle(
            "active",
            l.getAttribute("href") === "#" + id
          );
        });
      });
    },
    { rootMargin: "0px 0px -80% 0px", threshold: 0 }
  );

  // Start observing every section that has an ID inside the tab panels
  panels.forEach(function (panel) {
    panel.querySelectorAll("section[id]").forEach(function (sec) {
      observer.observe(sec);
    });
  });
});
