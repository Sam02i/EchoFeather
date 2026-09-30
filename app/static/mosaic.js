(() => {
  'use strict';
  const board = document.getElementById('mosaic');
  // Grout centerlines measured in the original 1410 × 864 tile image.
  const xEdges = [0,56,109,170,229,288,346,403,459,514,569,624,678,728,787,841,897,953,1010,1062,1123,1181,1239,1297,1354,1410];
  const yEdges = [0,65,125,184,244,302,360,418,477,534,592,652,712,774,836,864];
  const cols = xEdges.length - 1, rows = yEdges.length - 1, tiles = [];
  let scale = 1, offsetX = 0, offsetY = 0;
  let width = 0, height = 0, previous = null, frame = 0, point = null;
  const fragment = document.createDocumentFragment();
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const tile = document.createElement('div');
      tile.className = 'tile';
      tile.innerHTML = '<div class="flipper"><div class="face front"></div><div class="edge"></div><div class="face back"></div></div>';
      tile.setAttribute('aria-hidden', 'true');
      tile.addEventListener('transitionend', () => tile.classList.remove('moving'));
      fragment.appendChild(tile);
      tiles.push({ element: tile, x, y, revealed: false });
    }
  }
  board.appendChild(fragment);
  function layout() {
    width = board.clientWidth;
    height = board.clientHeight;
    // Keep both images aligned, preserving their proportions like object-fit: cover.
    scale = Math.max(width / 1410, height / 864);
    const imageWidth = 1410 * scale, imageHeight = 864 * scale;
    offsetX = (width - imageWidth) / 2; offsetY = (height - imageHeight) / 2;
    tiles.forEach(({ element, x, y }) => {
      const left = offsetX + xEdges[x] * scale, top = offsetY + yEdges[y] * scale;
      Object.assign(element.style, { left: `${left}px`, top: `${top}px`, width: `${(xEdges[x + 1] - xEdges[x]) * scale + .2}px`, height: `${(yEdges[y + 1] - yEdges[y]) * scale + .2}px` });
      element.querySelectorAll('.face').forEach(face => {
        face.style.backgroundSize = `${imageWidth}px ${imageHeight}px`;
        face.style.backgroundPosition = `${offsetX - left}px ${offsetY - top}px`;
      });
    });
    previous = null;
  }
  function reveal(tile, value = true) {
    if (tile.revealed === value) return;
    tile.revealed = value;
    tile.element.classList.add('moving');
    tile.element.classList.toggle('revealed', value);
  }
  function brush(px, py) {
    const sx = (px - offsetX) / scale, sy = (py - offsetY) / scale;
    const x = xEdges.findIndex((edge, i) => i < cols && sx >= edge && sx < xEdges[i + 1]);
    const y = yEdges.findIndex((edge, i) => i < rows && sy >= edge && sy < yEdges[i + 1]);
    if (x >= 0 && y >= 0) reveal(tiles[y * cols + x]);
  }
  function paint() {
    frame = 0;
    if (!point) return;
    const from = previous || point;
    const dx = point.x - from.x, dy = point.y - from.y;
    // Visit only cells crossed by the pointer, including fast diagonal drags.
    const cuts = [0, 1];
    if (dx) xEdges.forEach(edge => {
      const t = (offsetX + edge * scale - from.x) / dx;
      if (t > 0 && t < 1) cuts.push(t);
    });
    if (dy) yEdges.forEach(edge => {
      const t = (offsetY + edge * scale - from.y) / dy;
      if (t > 0 && t < 1) cuts.push(t);
    });
    cuts.sort((a, b) => a - b);
    for (let i = 1; i < cuts.length; i++) {
      const t = (cuts[i - 1] + cuts[i]) / 2;
      brush(from.x + dx * t, from.y + dy * t);
    }
    brush(point.x, point.y);
    previous = point;
  }
  function move(event) {
    if (event.pointerType === 'touch' && event.type === 'pointermove' && !event.buttons) return;
    const rect = board.getBoundingClientRect();
    point = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    if (!frame) frame = requestAnimationFrame(paint);
  }
  board.addEventListener('pointermove', move);
  board.addEventListener('pointerdown', event => { previous = null; board.setPointerCapture(event.pointerId); move(event); });
  board.addEventListener('pointerleave', () => { previous = null; });
  board.addEventListener('pointercancel', () => { previous = null; });
  board.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); tiles.forEach(tile => reveal(tile)); }
    if (event.key === 'Escape') { tiles.forEach(tile => reveal(tile, false)); previous = null; }
  });
  new ResizeObserver(layout).observe(board);
  layout();
})();
