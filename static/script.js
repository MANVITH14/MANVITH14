const form = document.getElementById('upload-form');
const imageInput = document.getElementById('image-input');
const preview = document.getElementById('preview');
const previewImg = document.getElementById('preview-img');
const result = document.getElementById('result');
const annotatedImg = document.getElementById('annotated-img');
const facesDiv = document.getElementById('faces');
const potentialMatchesDiv = document.getElementById('potential-matches');
const errorDiv = document.getElementById('error');

function showError(message) {
  errorDiv.textContent = message;
  errorDiv.classList.remove('hidden');
  result.classList.add('hidden');
  preview.classList.add('hidden');
}

function hideError() {
  errorDiv.classList.add('hidden');
}

function displayMatches(matches) {
  if (!matches || matches.length === 0) {
    if (potentialMatchesDiv) {
      potentialMatchesDiv.innerHTML = `
        <div style="padding: 1rem; background: #f0fdf4; border-left: 4px solid #10b981; border-radius: 8px;">
          <h4 style="color: #065f46; margin: 0 0 0.5rem 0;">✅ No Missing Person Matches</h4>
          <p style="color: #047857; margin: 0;">No potential matches found in the missing persons database.</p>
        </div>
      `;
    }
    return;
  }

  const matchesHTML = matches.map(match => {
    const confidenceClass = match.confidence >= 80 ? 'high-confidence' : 
                          match.confidence >= 60 ? 'medium-confidence' : 'low-confidence';
    
    const borderColor = match.confidence >= 80 ? '#ef4444' : 
                       match.confidence >= 60 ? '#f59e0b' : '#6b7280';
    
    return `
      <div style="padding: 1.5rem; background: white; border-left: 4px solid ${borderColor}; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
          <div>
            <h4 style="color: #1f2937; margin: 0 0 0.5rem 0; font-size: 1.25rem;">
              🚨 POTENTIAL MATCH: ${match.missing_person.name}
            </h4>
            <div style="color: #6b7280; font-size: 0.875rem;">
              Reported: ${new Date(match.missing_person.date_reported).toLocaleDateString()}
            </div>
          </div>
          <div style="text-align: right;">
            <div style="background: ${borderColor}; color: white; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; font-size: 0.875rem;">
              ${match.confidence}% Match
            </div>
          </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
          <div>
            <strong style="color: #374151;">Age:</strong>
            <div style="color: #6b7280;">${match.missing_person.age || 'Unknown'}</div>
          </div>
          <div>
            <strong style="color: #374151;">Gender:</strong>
            <div style="color: #6b7280;">${match.missing_person.gender}</div>
          </div>
          ${match.missing_person.last_seen_location ? `
            <div>
              <strong style="color: #374151;">Last Seen:</strong>
              <div style="color: #6b7280;">${match.missing_person.last_seen_location}</div>
            </div>
          ` : ''}
        </div>
        
        ${match.missing_person.description ? `
          <div style="margin-bottom: 1rem;">
            <strong style="color: #374151;">Description:</strong>
            <div style="color: #6b7280; margin-top: 0.25rem;">${match.missing_person.description}</div>
          </div>
        ` : ''}
        
        <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
          <strong style="color: #374151; display: block; margin-bottom: 0.5rem;">Match Analysis:</strong>
          <div style="font-size: 0.875rem; color: #6b7280;">
            ${match.match_reasons.gender_match ? '✅' : '❌'} Gender match<br>
            ${match.match_reasons.age_match ? '✅' : '❌'} Age compatibility (${match.match_reasons.age_score}% score)
          </div>
        </div>
        
        <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
          <a href="/missing-person/${match.missing_person.id}" 
             style="background: #3b82f6; color: white; padding: 0.5rem 1rem; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 0.875rem;">
            View Full Details
          </a>
          <button onclick="reportSighting(${match.missing_person.id}, ${match.confidence})"
                  style="background: #10b981; color: white; padding: 0.5rem 1rem; border-radius: 8px; border: none; font-weight: 600; font-size: 0.875rem; cursor: pointer;">
            Report Sighting
          </button>
          ${match.missing_person.contact_phone || match.missing_person.contact_email ? `
            <button onclick="showContactInfo('${match.missing_person.name}', '${match.missing_person.contact_phone || ''}', '${match.missing_person.contact_email || ''}')"
                    style="background: #f59e0b; color: white; padding: 0.5rem 1rem; border-radius: 8px; border: none; font-weight: 600; font-size: 0.875rem; cursor: pointer;">
              Contact Reporter
            </button>
          ` : ''}
        </div>
      </div>
    `;
  }).join('');

  if (potentialMatchesDiv) {
    potentialMatchesDiv.innerHTML = `
      <div style="margin-bottom: 1rem;">
        <h3 style="color: #ef4444; margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.5rem;">
          🚨 POTENTIAL MISSING PERSON MATCHES (${matches.length})
        </h3>
        <div style="background: #fef2f2; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #fecaca;">
          <strong style="color: #991b1b;">IMPORTANT:</strong> 
          <span style="color: #7f1d1d;">If you recognize any of these individuals, please contact the authorities immediately. These are AI-generated matches and require human verification.</span>
        </div>
      </div>
      ${matchesHTML}
    `;
  }
}

function reportSighting(personId, confidence) {
  const message = `Report sighting for person ID ${personId} with ${confidence}% confidence match?`;
  if (confirm(message)) {
    alert('In a real system, this would:\n\n• Create a sighting record\n• Notify authorities\n• Send alerts to family\n• Log GPS location\n• Request additional verification');
  }
}

function showContactInfo(name, phone, email) {
  let contactInfo = `Contact information for ${name}:\n\n`;
  if (phone) contactInfo += `Phone: ${phone}\n`;
  if (email) contactInfo += `Email: ${email}\n`;
  contactInfo += `\nPlease contact them if you have information about this missing person.`;
  alert(contactInfo);
}

if (imageInput) {
  imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        previewImg.src = e.target.result;
        preview.classList.remove('hidden');
        result.classList.add('hidden');
        hideError();
      };
      reader.readAsDataURL(file);
    }
  });
}

if (form) {
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const file = imageInput.files[0];
    if (!file) {
      showError('Please select an image file first.');
      return;
    }

    const formData = new FormData();
    formData.append('image', file);

    // Show loading state
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.textContent = 'Analyzing...';
    submitBtn.disabled = true;

    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Analysis failed');
      }

      // Display results
      annotatedImg.src = data.annotated_image;
      
      // Display face detection results
      if (data.faces && data.faces.length > 0) {
        const facesHTML = data.faces.map((face, index) => `
          <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h4 style="color: #1f2937; margin: 0 0 0.5rem 0;">Face ${index + 1} (Demo Detection)</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 0.5rem; font-size: 0.875rem;">
              <div><strong>Gender:</strong> ${face.gender} (${Math.round(face.gender_confidence * 100)}%)</div>
              <div><strong>Age:</strong> ${face.age_range} (${Math.round(face.age_confidence * 100)}%)</div>
              <div><strong>Position:</strong> ${face.box.x}, ${face.box.y}</div>
              <div><strong>Size:</strong> ${face.box.w} × ${face.box.h}</div>
            </div>
            <div style="background: #fef3c7; padding: 0.5rem; border-radius: 4px; margin-top: 0.5rem; font-size: 0.75rem; color: #92400e;">
              🔬 Demo Mode: This is simulated AI detection for demonstration purposes
            </div>
          </div>
        `).join('');
        facesDiv.innerHTML = facesHTML;
      } else {
        facesDiv.innerHTML = '<p style="color: #6b7280; font-style: italic;">No faces detected in the image.</p>';
      }

      // Display potential matches
      displayMatches(data.potential_matches);

      result.classList.remove('hidden');
      hideError();

    } catch (error) {
      console.error('Analysis error:', error);
      showError(`Analysis failed: ${error.message}`);
    } finally {
      // Restore button state
      submitBtn.textContent = originalText;
      submitBtn.disabled = false;
    }
  });
}