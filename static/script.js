let mediaRecorder;
let chunks = [];

const recordBtn = document.getElementById('recordBtn');

recordBtn.onclick = async function () {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') {
    chunks = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = e => chunks.push(e.data);
    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunks, { type: 'audio/wav' });
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64Audio = reader.result;
        const response = await fetch('/voice-input', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ audio: base64Audio })
        });
        const result = await response.json();
        if (result.query) {
          document.getElementById('query_text').value = result.query;
        } else {
          alert('Voice recognition failed: ' + result.error);
        }
      };
      reader.readAsDataURL(blob);
    };

    mediaRecorder.start();
    recordBtn.textContent = '⏹️ Stop Recording';
  } else if (mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    recordBtn.textContent = '🎙️ Record Voice Query';
  }
};
