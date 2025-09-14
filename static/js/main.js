document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadForm = document.getElementById('upload-form');
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const changeImageBtn = document.getElementById('change-image-btn');
    const submitBtn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const resultImage = document.getElementById('result-image');
    const diseaseName = document.getElementById('disease-name');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceBar = document.getElementById('confidence-bar');
    const diseaseDescription = document.getElementById('disease-description');
    const treatmentRecommendation = document.getElementById('treatment-recommendation');
    const loadingContainer = document.getElementById('loading-container');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');

    // Event Listeners
    browseBtn.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('click', () => fileInput.click());
    changeImageBtn.addEventListener('click', () => fileInput.click());
    
    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        uploadArea.classList.add('dragover');
    }
    
    function unhighlight() {
        uploadArea.classList.remove('dragover');
    }
    
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        handleFiles(files);
    }
    
    // Handle file selection
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            handleFiles(this.files);
        }
    });
    
    function handleFiles(files) {
        const file = files[0];
        if (file && isValidFile(file)) {
            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                resultImage.src = e.target.result;
                uploadArea.classList.add('d-none');
                previewContainer.classList.remove('d-none');
                submitBtn.disabled = false;
                
                // Hide previous results if any
                resultContainer.classList.add('d-none');
                errorContainer.classList.add('d-none');
            };
            reader.readAsDataURL(file);
        } else {
            showError('Please select a valid image file (JPG, JPEG, or PNG).');
        }
    }
    
    function isValidFile(file) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        return validTypes.includes(file.type);
    }
    
    // Form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!fileInput.files || !fileInput.files[0]) {
            showError('Please select an image first.');
            return;
        }
        
        // Show loading
        loadingContainer.classList.remove('d-none');
        resultContainer.classList.add('d-none');
        errorContainer.classList.add('d-none');
        submitBtn.disabled = true;
        
        // Prepare form data
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        // Send request
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Something went wrong');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loading
            loadingContainer.classList.add('d-none');
            
            // Update result
            diseaseName.textContent = formatDiseaseName(data.class);
            confidenceValue.textContent = `${data.confidence}%`;
            confidenceBar.style.width = `${data.confidence}%`;
            diseaseDescription.textContent = data.description;
            treatmentRecommendation.textContent = data.treatment;
            
            // Show result
            resultContainer.classList.remove('d-none');
            resultContainer.scrollIntoView({ behavior: 'smooth' });
            submitBtn.disabled = false;
        })
        .catch(error => {
            // Hide loading
            loadingContainer.classList.add('d-none');
            showError(error.message);
            submitBtn.disabled = false;
        });
    });
    
    function showError(message) {
        errorMessage.textContent = message;
        errorContainer.classList.remove('d-none');
        errorContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    function formatDiseaseName(name) {
        // Replace underscores with spaces and format the disease name
        return name.split('___').map(part => {
            return part.split('_').map(word => {
                return word.charAt(0).toUpperCase() + word.slice(1);
            }).join(' ');
        }).join(' - ');
    }
});