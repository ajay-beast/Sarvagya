// app.js

async function sendMessage() {
  const userQuery = document.getElementById('user-query').value;
  
  if (userQuery.trim() === '') {
    alert('Please enter the prompt')
    return};

  // Get the selected JSON file
  const fileInput = document.getElementById('json-file');
  const file = fileInput.files[0];
  const sendButton = document.querySelector('.btn-primary');
  sendButton.disabled = true; 
    
  if (!file) {
    // alert('Please upload a JSON file.');
    const response = await fetch('http://127.0.0.1:5000/api/process-query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ history: '', query: userQuery }),
    });
    const responseData = await response.json();
    displayResponse(responseData);
    sendButton.disabled=false
  }

  else{
    // Read the content of the file
    const fileContent = await readFile(file);

    // Parse the JSON content
    const jsonContent = JSON.parse(fileContent);

    // Send user query and history to backend
    const response = await fetch('http://127.0.0.1:5000/api/process-query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ history: jsonContent, query: userQuery }),
    });

    const responseData = await response.json();
   // Display the AI response
    displayResponse(responseData);
    sendButton.disabled=false
  }
}

function displayResponse(responseData) {
  const responseContainer = document.getElementById('response-container');
  // const responseDiv = document.createElement('div');
  // responseDiv.className = 'alert alert-info';
  // responseDiv.textContent = responseData.message; // assuming responseData has a 'message' property
  // responseContainer.innerHTML = JSON.stringify(responseData.query); 
  responseContainer.innerHTML = responseData.res; // Clear previous responses
  // responseContainer.appendChild(responseDiv);
  console.log(responseData)
}

async function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (event) => {
      resolve(event.target.result);
    };

    reader.onerror = (error) => {
      reject(error);
    };

    reader.readAsText(file);
  });
}
