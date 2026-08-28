// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The only script on the site: filter the sidebar. Every page is readable with
// JavaScript disabled, so this stays an enhancement rather than a dependency.

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
