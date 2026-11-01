// static/js/script.js
// Complete JavaScript with Theme Toggle

// ========== INITIALIZATION ==========
document.addEventListener('DOMContentLoaded', function() {
    // Initialize AOS if available
    if (typeof AOS !== 'undefined') {
        AOS.init({
            duration: 800,
            once: true,
            offset: 100
        });
    }
    
    // Initialize theme toggle
    initThemeToggle();
    
    // Initialize navbar scroll effect
    initNavbarScroll();
    
    // Animate stats on dashboard
    animateStats();
});

// ========== THEME TOGGLE ==========
function initThemeToggle() {
    const toggle = document.querySelector('.theme-toggle');
    if (!toggle) return;
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
        toggle.innerHTML = '<i class="fas fa-sun"></i>';
    } else {
        document.body.classList.remove('light-theme');
        toggle.innerHTML = '<i class="fas fa-moon"></i>';
    }
    
    // Toggle theme on click
    toggle.addEventListener('click', function() {
        const isLight = document.body.classList.toggle('light-theme');
        
        // Update icon with animation
        if (isLight) {
            this.innerHTML = '<i class="fas fa-sun"></i>';
            localStorage.setItem('theme', 'light');
            showToast('Light mode activated', 'info');
        } else {
            this.innerHTML = '<i class="fas fa-moon"></i>';
            localStorage.setItem('theme', 'dark');
            showToast('Dark mode activated', 'info');
        }
    });
}

// ========== NAVBAR SCROLL EFFECT ==========
function initNavbarScroll() {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;
    
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
}

// ========== ANIMATE STATS ==========
function animateStats() {
    const statValues = document.querySelectorAll('.stat-value[data-count]');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const element = entry.target;
                const target = parseInt(element.getAttribute('data-count'));
                let current = 0;
                const increment = target / 50;
                
                const timer = setInterval(() => {
                    current += increment;
                    if (current >= target) {
                        element.textContent = target;
                        clearInterval(timer);
                    } else {
                        element.textContent = Math.floor(current);
                    }
                }, 30);
                
                observer.unobserve(element);
            }
        });
    }, { threshold: 0.5 });
    
    statValues.forEach(stat => observer.observe(stat));
}

// ========== FILE UPLOAD FUNCTIONS ==========
function setupFileUpload() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    
    if (!uploadZone) return;
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = '#6366f1';
        uploadZone.style.transform = 'scale(1.01)';
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.style.borderColor = 'var(--border-color)';
        uploadZone.style.transform = 'scale(1)';
    });
    
    uploadZone.addEventListener('drop', async (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'var(--border-color)';
        uploadZone.style.transform = 'scale(1)';
        
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.csv')) {
            await uploadFile(file);
        } else {
            showToast('Please upload a valid CSV file', 'error');
        }
    });
    
    if (fileInput) {
        fileInput.addEventListener('change', async (e) => {
            if (e.target.files[0]) {
                await uploadFile(e.target.files[0]);
            }
        });
    }
}

function triggerFileInput() {
    document.getElementById('fileInput')?.click();
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const resultDiv = document.getElementById('analysisResult');
    const chartsContainer = document.getElementById('chartsContainer');
    
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="loading-spinner"></div><p class="text-center">Analyzing your data...</p>';
        resultDiv.style.display = 'block';
    }
    
    if (chartsContainer) chartsContainer.style.display = 'none';
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.success) {
            displayAnalysis(data.analysis);
            displayCharts(data.charts);
            showToast(`Successfully analyzed ${data.filename}`, 'success');
        } else {
            if (resultDiv) {
                resultDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            }
            showToast(data.error, 'error');
        }
    } catch (error) {
        if (resultDiv) {
            resultDiv.innerHTML = `<div class="alert alert-danger">Upload failed: ${error.message}</div>`;
        }
        showToast(error.message, 'error');
    }
}

function displayAnalysis(analysis) {
    const container = document.getElementById('analysisResult');
    if (!container) return;
    
    let html = `
        <div class="analysis-card">
            <h3><i class="fas fa-info-circle"></i> File Information</h3>
            <div class="info-grid">
                <div class="info-item"><strong>Filename:</strong> ${analysis.basic_info.filename}</div>
                <div class="info-item"><strong>Rows:</strong> ${analysis.basic_info.rows.toLocaleString()}</div>
                <div class="info-item"><strong>Columns:</strong> ${analysis.basic_info.columns}</div>
                <div class="info-item"><strong>Memory:</strong> ${analysis.basic_info.memory_usage}</div>
                <div class="info-item"><strong>Data Quality:</strong> ${analysis.data_quality}%</div>
                <div class="info-item"><strong>Type:</strong> ${analysis.dataset_type}</div>
            </div>
            
            <h3><i class="fas fa-table"></i> Column Analysis</h3>
            <div class="table-responsive">
                <table class="data-table">
                    <thead>
                        <tr><th>Column</th><th>Type</th><th>Null %</th><th>Unique</th><th>Min/Max</th></tr>
                    </thead>
                    <tbody>
                        ${analysis.columns.map(col => `
                            <tr>
                                <td><code>${col.name}</code></td>
                                <td>${col.type}</td>
                                <td><span class="${col.null_percentage === 0 ? 'badge-success' : 'badge-warning'}">${col.null_percentage}%</span></td>
                                <td>${col.unique_values}</span></td>
                                <td>${col.min !== undefined ? `${col.min.toFixed(0)} - ${col.max.toFixed(0)}` : '-'}</span></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

function displayCharts(charts) {
    const container = document.getElementById('chartsContainer');
    if (!container) return;
    
    let html = '<div class="charts-grid">';
    
    const titles = {
        'distributions': 'Data Distributions',
        'correlation': 'Correlation Heatmap',
        'missing_values': 'Missing Values Pattern',
        'boxplots': 'Outlier Detection'
    };
    
    for (const [key, chartBase64] of Object.entries(charts)) {
        html += `
            <div class="chart-card">
                <h4><i class="fas fa-chart-line"></i> ${titles[key] || key}</h4>
                <img src="data:image/png;base64,${chartBase64}" alt="${key}" onclick="window.open(this.src)">
            </div>
        `;
    }
    
    html += '</div>';
    container.innerHTML = html;
    container.style.display = 'block';
}

// ========== LOAD SAMPLE DATA ==========
async function loadSampleData(type) {
    try {
        const response = await fetch('/api/get-sample-data');
        const data = await response.json();
        
        if (type === 'loan' && data.loan) {
            document.getElementById('monthly_income').value = data.loan.monthly_income;
            document.getElementById('credit_score').value = data.loan.credit_score;
            document.getElementById('limit_bal').value = data.loan.LIMIT_BAL || 200000;
            document.getElementById('age').value = data.loan.AGE || 35;
            document.getElementById('employment_years').value = data.loan.employment_years;
            showToast('Sample loan data loaded!', 'success');
        } else if (type === 'investment' && data.investment) {
            document.getElementById('amount').value = data.investment.amount;
            document.getElementById('time_horizon').value = data.investment.time_horizon;
            document.getElementById('risk_appetite').value = data.investment.risk_appetite;
            showToast('Sample investment data loaded!', 'success');
        }
    } catch(e) {
        console.error(e);
        showToast('Could not load sample data', 'error');
    }
}

// ========== RESET FORMS ==========
function resetForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return;
    
    form.reset();
    
    // Reset specific fields
    if (formId === 'loanForm') {
        document.getElementById('monthly_income').value = '75000';
        document.getElementById('credit_score').value = '720';
        document.getElementById('limit_bal').value = '200000';
        document.getElementById('age').value = '35';
        document.getElementById('employment_years').value = '5';
        document.getElementById('education').value = '2';
        document.getElementById('pay_0').value = '0';
        document.getElementById('loan_term_years').value = '5';
    } else if (formId === 'investmentForm') {
        document.getElementById('amount').value = '100000';
        document.getElementById('time_horizon').value = '10';
        document.getElementById('risk_appetite').value = 'Moderate';
    }
    
    // Hide result divs
    const resultDivs = ['predictionResult', 'investmentResult', 'riskResult'];
    resultDivs.forEach(id => {
        const div = document.getElementById(id);
        if (div) div.style.display = 'none';
    });
    
    showToast('Form reset', 'info');
}

// ========== TOAST NOTIFICATION ==========
function showToast(message, type = 'info') {
    let toastContainer = document.querySelector('.toast-container');
    
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }
    
    const toast = document.createElement('div');
    toast.className = `toast-notification toast-${type}`;
    toast.innerHTML = `<i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i> ${message}`;
    
    toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ========== LOAD DATASETS INFO ==========
async function loadDatasetsInfo() {
    try {
        const response = await fetch('/api/get-datasets-info');
        const data = await response.json();
        const container = document.getElementById('datasetList');
        
        if (container) {
            let html = '<div class="row">';
            for (const [name, info] of Object.entries(data)) {
                if (info.exists) {
                    html += `
                        <div class="col-md-3 col-sm-6 mb-3">
                            <div class="dataset-badge">
                                <i class="fas fa-chart-line"></i>
                                ${name.replace(/_/g, ' ').toUpperCase()}
                                <small>${info.size_mb} MB | ${info.columns} cols</small>
                            </div>
                        </div>
                    `;
                }
            }
            html += '</div>';
            container.innerHTML = html;
        }
    } catch(e) {
        console.error(e);
    }
}

// Add slideOut animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);