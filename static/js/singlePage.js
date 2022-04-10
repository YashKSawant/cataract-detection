document.getElementById("uploadFile").addEventListener("click", uploadImage);

async function uploadImage(e) {
    const form_data = new FormData();
	const ins = document.getElementById('fileInput').files.length;
    console.log(form_data, ins, document.getElementById('fileInput').files)
    for (let x = 0; x < ins; x++) {
        console.log(document.getElementById('fileInput').files[x])
        form_data.append("files[]", document.getElementById('fileInput').files[x]);
    }
    console.log(form_data)

    const response = await fetch("/uploadImage", {
        method: 'POST', 
        body: form_data
      });
      console.log(response.json())
}