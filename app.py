from flask import Flask, request, redirect, flash, render_template, url_for
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Load the trained model


model_path = 'model/srcnn_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match model's expected input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def process_image(file_path):
    image = Image.open(file_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)
@app.route('/')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the image
        input_image = process_image(file_path)

        # Inference with the model
        with torch.no_grad():
            output = model(input_image)

        # Post-process the output
        output_image = output.squeeze().permute(1, 2, 0).cpu().numpy()
        output_image = (output_image * 255).clip(0, 255).astype('uint8')
        output_image = Image.fromarray(output_image)

        # Save the output image
        output_name = file.filename.split('.')[0] + '_output.png'
        output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_name)
        output_image.save(output_image_path)

        flash('Image successfully uploaded and processed!')
        return render_template('result.html',
                               input_image=url_for('static', filename='uploads/' + file.filename),
                               output_image=url_for('static', filename='uploads/' + output_name))



os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if __name__ == "__main__":
    app.run(debug=True)
