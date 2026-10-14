const range = document.getElementById('compare-range');
const after = document.getElementById('compare-after-wrap');
const handle = document.getElementById('compare-handle');
const beforeImage = document.getElementById('compare-before');
const afterImage = document.getElementById('compare-after');
const caption = document.getElementById('scene-caption');

function updateComparison() {
  const value = Number(range.value);
  after.style.clipPath = `inset(0 0 0 ${value}%)`;
  handle.style.left = `${value}%`;
  range.setAttribute('aria-valuetext', `${value}% clear image revealed`);
}
range.addEventListener('input', updateComparison);
updateComparison();

for (const button of document.querySelectorAll('[data-scene]')) {
  button.addEventListener('click', () => {
    const scene = button.dataset.scene;
    beforeImage.src = `./assets/scene-${scene}-clear.jpg`;
    afterImage.src = `./assets/scene-${scene}-cloudy.jpg`;
    beforeImage.alt = `Clear satellite view of sample ${scene}`;
    afterImage.alt = `The same sample ${scene} with a PGCS-generated cloud`;
    caption.textContent = `Paired author-provided Landsat 8/9 example · sample ${scene}. Drag the divider to compare identical ground content.`;
    for (const item of document.querySelectorAll('[data-scene]')) {
      const active = item === button;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-pressed', String(active));
    }
  });
}

for (const button of document.querySelectorAll('[data-series]')) {
  button.addEventListener('click', () => {
    const series = button.dataset.series;
    const sceneName = series === '1' ? 'A' : 'B';
    for (const level of ['light', 'medium', 'dense']) {
      const image = document.getElementById(`density-${level}`);
      image.src = `./assets/thickness-${series}-${level}.jpg`;
      image.alt = `${level[0].toUpperCase()}${level.slice(1)} simulated cloud over scene ${sceneName}`;
    }
    for (const item of document.querySelectorAll('[data-series]')) {
      const active = item === button;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-pressed', String(active));
    }
  });
}

const copyButton = document.getElementById('copy-citation');
copyButton.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(document.getElementById('bibtex').textContent);
    copyButton.textContent = 'Copied';
    setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; }, 2000);
  } catch {
    copyButton.textContent = 'Select text to copy';
  }
});
