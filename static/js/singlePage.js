document.getElementById('uploadFile').addEventListener('click', uploadImage);
document.getElementById('fileInput').addEventListener('change', selectImage);

function selectImage(e) {
  document.getElementById('inputImage').src= URL.createObjectURL(document.getElementById('fileInput').files[0]);
}

async function uploadImage(e) {
  document.getElementsByClassName("loader")[0].style.display = "block";
  const form_data = new FormData();
  const ins = document.getElementById('fileInput').files.length;
  for (let x = 0; x < ins; x++) {    
    form_data.append('files[]', document.getElementById('fileInput').files[x]);
  }

  const response = await fetch('/uploadImage', {
    method: 'POST',
    body: form_data,
  });

  const jsonData = await response.json();
  document.getElementsByClassName("result-image")[0].src = document.getElementById('inputImage').src;
  document.getElementById('body').innerHTML = jsonData.result[0]==1 ? "Predicted Result: Cataract" : "Predicted Result: Normal";
  document.getElementsByClassName("loader")[0].style.display = "none";
}
