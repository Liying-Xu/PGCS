const copyButton = document.getElementById('copy-citation');
const bibtex = document.getElementById('bibtex');

document.querySelectorAll('.scene-compare input[type="range"]').forEach(slider => {
  const update = () => {
    slider.parentElement.style.setProperty('--split', `${slider.value}%`);
    slider.setAttribute('aria-valuetext', `${slider.value}% clear, ${100 - Number(slider.value)}% PGCS synthesis`);
  };
  slider.addEventListener('input', update);
  update();
});

copyButton.addEventListener('click', async () => {
  const citation = bibtex.textContent;
  try {
    await navigator.clipboard.writeText(citation);
  } catch {
    const field = document.createElement('textarea');
    field.value = citation;
    document.body.appendChild(field);
    field.select();
    document.execCommand('copy');
    field.remove();
  }
  copyButton.textContent = 'Copied';
  setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; }, 2000);
});
