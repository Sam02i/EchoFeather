const content = document.getElementById('content');
const route = location.pathname.replace(/\/$/, '') || '/';
document.title = `${({'/':'Home','/bird':'Bird','/audio':'Audio','/about':'About','/collection':'Collection'})[route] || 'Bird ID'} · Samiksha`;
document.querySelectorAll('nav a').forEach(link => {
  if (link.getAttribute('href') === route) link.setAttribute('aria-current', 'page');
});
const escapeText = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function uploadForm(kind) {
  const photo = kind === 'image';
  return `<form id="identify-form" data-kind="${kind}">
    <div class="panel-heading"><h2>${photo ? 'Upload a photograph' : 'Upload a recording'}</h2></div>
    <label class="dropzone" for="file">
      <strong>${photo ? 'Choose a bird photograph' : 'Choose a bird call'}</strong>
      <span class="dropzone-note">Drag and drop, or browse your files</span>
      <span class="browse-chip">${photo ? 'Browse photos' : 'Browse recordings'}</span>
      <input class="file-input" id="file" name="file" type="file" accept="${photo ? 'image/jpeg,image/png,image/webp' : '.mp3,audio/mpeg,audio/mp3,.wav,audio/wav,.flac,audio/flac,.ogg,audio/ogg,.m4a,audio/mp4'}" aria-label="${photo ? 'Choose a bird photograph' : 'Choose a bird call'}" required>
    </label>
    <p id="file-name" class="file-name" hidden></p>
    <p class="hint">${photo ? 'JPEG, PNG or WebP · up to 25 MB' : 'MP3, WAV, FLAC, OGG or M4A · up to 25 MB<br>3 seconds minimum · first 30 seconds analyzed'}</p>
    <button class="primary" type="submit"><span>${photo ? 'Identify bird' : 'Identify call'}</span></button>
    <p class="upload-tip">${photo ? 'A clear photo with one bird works best.' : 'A clear call with less background noise works best.'}</p>
    <p id="message" class="message" role="status" aria-live="polite"></p>
  </form>`;
}

function discoveryPage(kind) {
  const photo = kind === 'image';
  return `<section class="discovery-page">
    <div class="discovery-heading"><p class="eyebrow">${photo ? 'Visual identification' : 'Acoustic identification'}</p>
      <h1>${photo ? 'Bird identification.' : 'Birdsong identification.'}</h1>
      <p class="intro">${photo ? 'Upload a photograph to compare possible species.' : 'Upload a bird call to compare possible species.'}</p>
    </div>
    <div class="discovery-layout">
      <div class="preview-column">
        <figure class="nature-card">
          <div class="side-media">
            <img id="side-image" src="/static/assets/${photo ? 'garden-birds.jpeg' : 'bird.png'}" alt="${photo ? 'Two blue tits perched among pink blossoms' : 'Green and gold bird mosaic'}">
            ${photo ? '' : '<div id="audio-preview" class="audio-preview" hidden><span id="audio-caption">Selected recording</span><audio id="preview" controls></audio></div>'}
          </div>
          <figcaption id="media-caption">${photo ? 'Photographs / Species recognition' : 'Recordings / Call recognition'}</figcaption>
        </figure>
        <div id="results" class="results" aria-live="polite"></div>
      </div>
      <section class="identify-panel" aria-label="${photo ? 'Identify a bird photograph' : 'Identify a bird call'}">${uploadForm(kind)}</section>
    </div>
  </section>`;
}

async function connectForm() {
  const form = document.getElementById('identify-form');
  const fileInput = document.getElementById('file');
  const photo = form.dataset.kind === 'image';
  const sideImage = document.getElementById('side-image');
  const originalImage = sideImage.src;
  const originalAlt = sideImage.alt;
  const preview = photo ? sideImage : document.getElementById('preview');
  const audioPreview = document.getElementById('audio-preview');
  const caption = document.getElementById('media-caption');
  const message = document.getElementById('message');
  const results = document.getElementById('results');
  const button = form.querySelector('button');
  let previewURL;
  let ready = true;
  function say(text, error = false) {
    message.textContent = text;
    message.classList.toggle('error', error);
  }
  fileInput.addEventListener('change', () => {
    if (previewURL) URL.revokeObjectURL(previewURL);
    results.replaceChildren();
    const filename = document.getElementById('file-name');
    filename.hidden = true;
    filename.textContent = '';
    if (!photo) { preview.pause(); preview.removeAttribute('src'); preview.load(); audioPreview.hidden = true; }
    sideImage.src = originalImage;
    sideImage.alt = originalAlt;
    sideImage.hidden = false;
    sideImage.classList.remove('uploaded');
    caption.textContent = photo ? 'Photographs / Species recognition' : 'Recordings / Call recognition';
    const file = fileInput.files[0];
    if (!file) return;
    if (file.size > 25 * 1024 * 1024) {
      say('Choose a file smaller than 25 MB.', true);
      fileInput.value = '';
      return;
    }
    filename.textContent = file.name;
    filename.hidden = false;
    previewURL = URL.createObjectURL(file);
    preview.src = previewURL;
    if (photo) {
      sideImage.classList.add('uploaded');
      sideImage.alt = 'Uploaded bird photograph';
      caption.textContent = file.name;
    } else {
      sideImage.hidden = true;
      audioPreview.hidden = false;
      document.getElementById('audio-caption').textContent = file.name;
      caption.textContent = 'Selected recording';
    }
    if (ready) say('');
  });
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (!fileInput.files[0] || !ready) return;
    button.disabled = true;
    form.classList.add('is-loading');
    fileInput.disabled = true;
    results.replaceChildren();
    say('Listening and identifying…');
    if (form.dataset.kind === 'image') say('Looking at your photograph…');
    const data = new FormData();
    data.append('file', fileInput.files[0]);
    try {
      const response = await fetch(`/api/predict/${form.dataset.kind}`, {method:'POST', body:data});
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Unable to identify this file. Please try again.');
      const predictions = [...result.predictions].sort((a, b) => b.score - a.score);
      const top = predictions[0];
      if (!top) throw new Error('No species match was returned. Please try another file.');
      results.innerHTML = `<article class="bird-summary">
        <p class="eyebrow">Highest-confidence match</p>
        <h2><strong>${escapeText(top.name)}</strong></h2>
        <p class="scientific-name">${escapeText(top.scientific_name)}</p>
        <dl class="bird-facts"><div><dt>Model confidence</dt><dd>${(top.score * 100).toFixed(1)}%</dd></div><div><dt>Identified from</dt><dd>${photo ? 'Photograph' : 'Bird call'}</dd></div></dl>
        <p class="results-note">This is the model’s closest match, not a confirmed identification.</p>
      </article>
      ${predictions.length > 1 ? `<details class="other-matches"><summary>Other possible matches</summary>${predictions.slice(1).map(item => `<div class="result"><div class="result-line"><span>${escapeText(item.name)}</span><span>${(item.score * 100).toFixed(1)}%</span></div><small>${escapeText(item.scientific_name)}</small></div>`).join('')}</details>` : ''}`;
      say('Identification complete. See the result beside your upload.');
      if (!photo) {
        // Audio has no photograph; show the matching species from the collection.
        try {
          const catalogResponse = await fetch('/api/species');
          if (catalogResponse.ok) {
            const birds = await catalogResponse.json();
            const bird = birds.find(item => item.species === top.species);
            if (bird?.image) {
              sideImage.src = bird.image;
              sideImage.alt = top.name;
              sideImage.hidden = false;
              caption.textContent = `${top.name} / Collection photograph`;
            }
          }
        } catch { /* The prediction and audio player still work without a catalog image. */ }
      }
    } catch (error) {
      say(error.message === 'Failed to fetch' ? 'Cannot reach the server. Please try again.' : error.message, true);
    } finally {
      form.classList.remove('is-loading');
      button.disabled = !ready;
      fileInput.disabled = false;
    }
  });
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('Status unavailable');
    const info = await response.json();
    const model = info[form.dataset.kind === 'image' ? 'vision' : 'audio'];
    ready = model.available;
    if (!ready) { button.disabled = true; say('This identification model is not ready yet. Please check back after it has been trained.', true); }
  } catch {
    say('The server is unavailable. Please restart the app and try again.', true);
  }
}

async function collection() {
  content.innerHTML = `<section class="page collection-page"><div class="catalog-head"><div><p class="eyebrow">The collection</p><h1>Species collection.</h1><p class="intro" id="catalog-summary">Loading species…</p></div><input id="search" class="search" type="search" placeholder="Search birds" aria-label="Search species"></div><div id="catalog" class="catalog"></div><p id="empty" class="empty" role="status"></p></section>`;
  try {
    const response = await fetch('/api/species');
    if (!response.ok) throw new Error('Could not load the collection. Please try again.');
    const birds = await response.json();
    document.getElementById('catalog-summary').textContent = `${birds.length} species supported by the identification models.`;
    function render() {
      const query = document.getElementById('search').value.toLowerCase().trim();
      const filtered = birds.filter(bird => `${bird.name} ${bird.scientific_name}`.toLowerCase().includes(query));
      document.getElementById('catalog').innerHTML = filtered.map(bird => `<article class="bird-card">
        ${bird.image ? `<img src="${escapeText(bird.image)}" alt="${escapeText(bird.name)}" loading="lazy">` : '<div class="photo-placeholder">Photograph unavailable</div>'}
        <div class="card-content"><h2>${escapeText(bird.name)}</h2><p><i>${escapeText(bird.scientific_name)}</i></p>${bird.photo?'<span class="tag">Photo</span>':''}${bird.audio?'<span class="tag">Audio</span>':''}</div></article>`).join('');
      document.getElementById('empty').textContent = filtered.length ? '' : 'No birds match your search.';
    }
    document.getElementById('search').addEventListener('input', render);
    render();
  } catch (error) {
    document.getElementById('catalog-summary').textContent = error.message;
  }
}

if (route === '/') {
  content.innerHTML = `<section class="home"><div id="mosaic" tabindex="0" role="img" aria-label="Interactive bird mosaic. Hover or drag to flip individual tiles. Press Enter to reveal the picture; Escape resets it."></div></section>`;
  const mosaic = document.createElement('script');
  mosaic.src = '/static/mosaic.js';
  document.body.appendChild(mosaic);
} else if (route === '/bird' || route === '/audio') {
  content.innerHTML = discoveryPage(route === '/bird' ? 'image' : 'audio');
  connectForm();
} else if (route === '/collection') {
  collection();
} else {
  content.innerHTML = `<section class="page about-page"><div class="discovery-heading"><p class="eyebrow">About the project</p><h1>About Samiksha.</h1><p class="intro">Bird identification through photographs and sound.</p></div><div class="about-layout"><figure class="nature-card"><img src="/static/assets/bird.png" alt="Green and gold bird mosaic"><figcaption>The Samiksha mosaic</figcaption></figure><div class="about-sections"><section><h2>Photo and audio identification</h2><p>Upload a photograph or recording to compare possible species. The photo model studies visual features; the audio model compares patterns in spectrograms—pictures of sound.</p></section><section><h2>Model limitations</h2><p>The models recognize a limited collection of species. Similar-looking birds, background noise and unsupported species can lead to incorrect suggestions. Confidence scores are model outputs, not proof of identification.</p></section><section><h2>Your uploads</h2><p>Photographs are processed in memory. Temporary recordings and spectrograms are removed after processing. Your uploads are not added to the training collection.</p></section><a href="/collection">View supported species</a></div></div></section>`;
}
