function getSelectedValue(name) {
    return document.querySelector(`input[name="${name}"]:checked`)?.value || '';
}

async function populateDropdown(elementId, category) {
    try {
        const response = await fetch(`/api/get_category_options?category=${category}`);
        const data = await response.json();
        
        if (data.status === 'success') {
            const select = document.getElementById(elementId);
            select.innerHTML = `<option value="" disabled selected>Select ${category}</option>`;
            data.options.forEach(option => {
                select.innerHTML += `<option value="${option}">${option}</option>`;
            });
        }
    } catch (error) {
        console.error(`Error loading ${category} options:`, error);
    }
}

async function onClickedEstimatePrice() {
    try {
        const inputData = {
            brand: document.getElementById('uiBrand').value,
            material: document.getElementById('uiMaterial').value,
            size: getSelectedValue('uiSize'),
            compartments: parseInt(document.getElementById('uiCompartments').value),
            laptop_compartment: getSelectedValue('uiLaptop'),
            waterproof: getSelectedValue('uiWaterproof'),
            style: document.getElementById('uiStyle').value,
            color: document.getElementById('uiColor').value,
            weight_capacity: parseFloat(document.getElementById('uiWeight').value)
        };

        const response = await fetch('/api/predict_price', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(inputData)
        });

        const result = await response.json();
        const resultDiv = document.getElementById('uiEstimatedPrice');

        if (result.status === 'success') {
            resultDiv.innerHTML = `
                <h2>Estimated Price</h2>
                <div class="price">$${result.estimated_price.toFixed(2)}</div>
            `;
            resultDiv.style.backgroundColor = '#e8f4f8';
        } else {
            resultDiv.innerHTML = `<div class="error">Error: ${result.error}</div>`;
            resultDiv.style.backgroundColor = '#f8d7da';
        }
    } catch (error) {
        console.error('Prediction failed:', error);
        document.getElementById('uiEstimatedPrice').innerHTML = `
            <div class="error">Network Error: Please try again</div>
        `;
    }
}

// Initialize form on load
window.onload = () => {
    ['Brand', 'Material', 'Style', 'Color'].forEach(category => {
        populateDropdown(`ui${category}`, category);
    });
};