// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Sidebar filtering, and the link back to the console. Every page is readable with
// JavaScript disabled, so both stay enhancements rather than dependencies.

// These pages are published to a static host and also served by the local server at
// /docs. Asking whether a HIP EP server answers at this origin is the same question as
// whether a link to the console would go anywhere, so the answer decides the link.
const studio = document.querySelector(".topnav .to-studio");
if (studio) {
  // Origin-absolute, like the link itself: the console and the API both live at the
  // server root, whatever depth the current page sits at.
  fetch("/v1/status", { method: "GET" })
    .then((response) => {
      if (response.ok) studio.hidden = false;
    })
    .catch(() => {
      // No server here: a static host, or the page opened straight off disk.
    });
}

const filter = document.querySelector(".sidebar .filter");
const nav = document.querySelector(".sidebar-nav");

if (filter && nav) {
  const links = [...nav.querySelectorAll("a")];
  const groups = [...nav.querySelectorAll("details")];

  filter.addEventListener("input", () => {
    const query = filter.value.trim().toLowerCase();

    for (const link of links) {
      const hit = !query || link.textContent.toLowerCase().includes(query);
      // Hiding the list item rather than the anchor keeps grouped items from
      // collapsing to a blank row.
      (link.closest("li") || link).hidden = !hit;
    }

    for (const group of groups) {
      const items = [...group.querySelectorAll("li")];
      const visible = items.filter((item) => !item.hidden).length;
      group.hidden = query !== "" && visible === 0;
      if (query) group.open = true;
    }
  });

  filter.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    filter.value = "";
    filter.dispatchEvent(new Event("input"));
  });
}
