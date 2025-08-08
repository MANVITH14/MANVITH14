const form = document.getElementById('upload-form');
const imageInput = document.getElementById('image-input');
const preview = document.getElementById('preview');
const previewImg = document.getElementById('preview-img');
const result = document.getElementById('result');
const annotatedImg = document.getElementById('annotated-img');
const facesDiv = document.getElementById('faces');
const errorDiv = document.getElementById('error');

imageInput.addEventListener('change', () => {
  facesDiv.innerHTML = '';
  errorDiv.classList.add('hidden');
  const file = imageInput.files[0];
  if (!file) {
    preview.classList.add('hidden');
    return;
  }
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    preview.classList.remove('hidden');
  };
  reader.readAsDataURL(file);
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = imageInput.files[0];
  if (!file) {
    return;
  }

  const data = new FormData();
  data.append('image', file);

  facesDiv.innerHTML = '';
  errorDiv.classList.add('hidden');
  result.classList.add('hidden');

  const submitBtn = form.querySelector('button[type="submit"]');
  submitBtn.disabled = true;
  submitBtn.textContent = 'Analyzing...';

  try {
    const res = await fetch('/analyze', { method: 'POST', body: data });
    const json = await res.json();
    if (!res.ok) {
      throw new Error(json.error || 'Request failed');
    }

    annotatedImg.src = json.annotated_image;
    result.classList.remove('hidden');

    if (!json.faces || json.faces.length === 0) {
      facesDiv.innerHTML = '<div class="face-card"><p>No faces detected.</p></div>';
    } else {
      json.faces.forEach((f, idx) => {
        const card = document.createElement('div');
        card.className = 'face-card';
        card.innerHTML = `
          <h4>Face ${idx + 1}</h4>
          <p><strong>Gender:</strong> ${f.gender} (${(f.gender_confidence * 100).toFixed(1)}%)</p>
          <p><strong>Age range:</strong> ${f.age_range} (${(f.age_confidence * 100).toFixed(1)}%)</p>
          <p><strong>Box:</strong> x=${f.box.x}, y=${f.box.y}, w=${f.box.w}, h=${f.box.h}</p>
        `;
        facesDiv.appendChild(card);
      });
    }
  } catch (err) {
    errorDiv.textContent = err.message;
    errorDiv.classList.remove('hidden');
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Analyze';
  }
});