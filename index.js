const webcamElement= document.getElementById('webcam');

//const net = undefined;

let isPredicting = false;

function startPredicting(){
	isPredicting=true;
	app();
}

function stopPredicting(){
	isPredicting=false;
	app();
}



async function app(){
	
  const svg = document.querySelector('svg');
      
  console.log('Loading model..');
	const net = await tf.automl.loadObjectDetection('model.json');
	console.log('Successfully loaded model');
	
	const webcam = await tf.data.webcam(webcamElement, {facingMode: 'environment'});
	while(isPredicting){  
  const img = await webcam.capture();
	const options = {score: 0.5, iou: 0.5, topk: 20};
  const predictions = await net.detect(img,options);
	
	console.log(predictions);
	
  predictions.forEach(prediction => {
    const {box, label, score} = prediction;
    const {left, top, width, height} = box;
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('width', width);
    rect.setAttribute('height', height);
    rect.setAttribute('x', left);
    rect.setAttribute('y', top);
    rect.setAttribute('class', 'box');
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', left + width / 2);
    text.setAttribute('y', top);
    text.setAttribute('dy', 12);
    text.setAttribute('class', 'label');
    text.textContent = `${label}: ${score.toFixed(3)}`;
    svg.appendChild(rect);
    svg.appendChild(text);
    const textBBox = text.getBBox();
    const textRect =
        document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    textRect.setAttribute('x', textBBox.x);
    textRect.setAttribute('y', textBBox.y);
    textRect.setAttribute('width', textBBox.width);
    textRect.setAttribute('height', textBBox.height);
    textRect.setAttribute('class', 'label-rect');
    svg.insertBefore(textRect, text);
  });

  console.log('Prediction made and graphed');
    
	img.dispose();
    
  await tf.nextFrame();
 // d3.selectAll("svg > *").remove();  
	}
	
}
