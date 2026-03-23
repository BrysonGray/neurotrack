// Opacity control
function updateOpacity(rowId, value) {
    const opacity = value / 100;
    const maskElements = document.querySelectorAll(`[id^="mask_${rowId}"]`);
    maskElements.forEach(el => el.style.opacity = opacity);
    const valueDisplay = document.getElementById('value_' + rowId);
    if (valueDisplay) {
        valueDisplay.textContent = opacity.toFixed(2);
    }
}

// Zoom state tracking
const zoomStates = {};
const VIEWER_CACHE_BUST = Date.now();
const reviewRows = new Set();
const rowMetadata = new Map();
let reviewFilterEnabled = false;

function setDatasetCollapsed(datasetKey, isCollapsed, skipCollapseAllSync = false) {
    const datasetBlock = document.getElementById(`dataset_${datasetKey}`);
    if (datasetBlock) {
        datasetBlock.classList.toggle('dataset-collapsed', isCollapsed);
    }
    const datasetRows = document.getElementById(`dataset_rows_${datasetKey}`);
    if (datasetRows) {
        datasetRows.style.display = isCollapsed ? 'none' : '';
    }

    const datasetCollapseCheckbox = document.querySelector(
        `.dataset-collapse-checkbox[data-dataset-key="${datasetKey}"]`
    );
    if (datasetCollapseCheckbox) {
        datasetCollapseCheckbox.checked = isCollapsed;
    }

    if (!skipCollapseAllSync) {
        syncCollapseAllToggle();
    }
}

function toggleDatasetCollapse(datasetKey, isCollapsed) {
    setDatasetCollapsed(datasetKey, isCollapsed);
}

function syncCollapseAllToggle() {
    const collapseAllToggle = document.getElementById('collapse_all_toggle');
    const datasetCollapseCheckboxes = Array.from(document.querySelectorAll('.dataset-collapse-checkbox'));
    if (!collapseAllToggle || datasetCollapseCheckboxes.length === 0) {
        return;
    }

    const collapsedCount = datasetCollapseCheckboxes.filter(cb => cb.checked).length;
    collapseAllToggle.checked = collapsedCount === datasetCollapseCheckboxes.length;
    collapseAllToggle.indeterminate = collapsedCount > 0 && collapsedCount < datasetCollapseCheckboxes.length;
}

function toggleCollapseAll(isCollapsed) {
    const datasetCollapseCheckboxes = Array.from(document.querySelectorAll('.dataset-collapse-checkbox'));
    datasetCollapseCheckboxes.forEach(checkbox => {
        const datasetKey = checkbox.getAttribute('data-dataset-key');
        setDatasetCollapsed(datasetKey, isCollapsed, true);
    });
    syncCollapseAllToggle();
}

function csvEscape(value) {
    const text = String(value ?? '');
    if (text.includes(',') || text.includes('"') || text.includes('\n')) {
        return `"${text.replace(/"/g, '""')}"`;
    }
    return text;
}

function exportReviewedCsv() {
    const checkedRowIds = Array.from(document.querySelectorAll('.review-checkbox:checked'))
        .map(checkbox => checkbox.getAttribute('data-row-id'));

    if (checkedRowIds.length === 0) {
        alert('No rows are checked for review.');
        return;
    }

    const lines = [];

    checkedRowIds.forEach(rowId => {
        const meta = rowMetadata.get(rowId);
        if (!meta) {
            return;
        }
        lines.push(csvEscape(meta.full_file_path));
    });

    const csvContent = lines.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const link = document.createElement('a');
    link.href = url;
    link.download = `review_checked_files_${timestamp}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

function normalizeIdPart(value) {
    return String(value || '').replace(/\W+/g, '_');
}

function toggleRowReview(rowId, isChecked, skipApplyFilter = false) {
    if (isChecked) {
        reviewRows.add(rowId);
    } else {
        reviewRows.delete(rowId);
    }
    const rowElement = document.getElementById(`review_row_${rowId}`);
    if (rowElement) {
        rowElement.classList.toggle('review-checked', isChecked);
    }
    const rowCheckbox = document.getElementById(`review_checkbox_${rowId}`);
    const datasetKey = rowCheckbox ? rowCheckbox.getAttribute('data-dataset-key') : null;
    if (datasetKey) {
        syncDatasetReviewCheckbox(datasetKey);
    }
    if (!skipApplyFilter) {
        applyReviewFilter();
    }
}

function syncDatasetReviewCheckbox(datasetKey) {
    const rowCheckboxes = Array.from(
        document.querySelectorAll(`.review-checkbox[data-dataset-key="${datasetKey}"]`)
    );
    const datasetCheckbox = document.querySelector(
        `.dataset-review-checkbox[data-dataset-key="${datasetKey}"]`
    );
    if (!datasetCheckbox || rowCheckboxes.length === 0) {
        return;
    }

    const checkedCount = rowCheckboxes.filter(cb => cb.checked).length;
    datasetCheckbox.checked = checkedCount === rowCheckboxes.length;
    datasetCheckbox.indeterminate = checkedCount > 0 && checkedCount < rowCheckboxes.length;
}

function toggleDatasetReview(datasetKey, isChecked) {
    const rowCheckboxes = Array.from(
        document.querySelectorAll(`.review-checkbox[data-dataset-key="${datasetKey}"]`)
    );

    rowCheckboxes.forEach(checkbox => {
        checkbox.checked = isChecked;
        const rowId = checkbox.getAttribute('data-row-id');
        toggleRowReview(rowId, isChecked, true);
    });

    syncDatasetReviewCheckbox(datasetKey);
    applyReviewFilter();
}

function updateSectionVisibility() {
    document.querySelectorAll('.dataset-block').forEach(datasetBlock => {
        const rows = Array.from(datasetBlock.querySelectorAll('.qc-row'));
        const hasVisibleRows = rows.some(row => row.style.display !== 'none');
        datasetBlock.style.display = hasVisibleRows ? '' : 'none';
    });

    document.querySelectorAll('.section-block').forEach(sectionBlock => {
        const datasetBlocks = Array.from(sectionBlock.querySelectorAll('.dataset-block'));
        const hasVisibleDatasets = datasetBlocks.some(block => block.style.display !== 'none');
        sectionBlock.style.display = hasVisibleDatasets ? '' : 'none';
    });
}

function applyReviewFilter() {
    document.querySelectorAll('.qc-row').forEach(row => {
        const rowId = row.getAttribute('data-row-id');
        const shouldShow = !reviewFilterEnabled || reviewRows.has(rowId);
        row.style.display = shouldShow ? '' : 'none';
    });
    updateSectionVisibility();
}

function setReviewFilterEnabled(isEnabled) {
    reviewFilterEnabled = isEnabled;
    applyReviewFilter();
}

function initializeZoom(imageId, imageSrc, maskSrc) {
    if (zoomStates[imageId]) return; // Already initialized
    
    const imageStack = document.getElementById('stack_' + imageId);
    const baseImage = document.getElementById('img_' + imageId);
    const overlayImage = document.getElementById('mask_' + imageId);
    const homeButton = document.getElementById('home_' + imageId);
    const zoomInfo = document.getElementById('info_' + imageId);
    
    if (!imageStack || !baseImage) return;
    
    // Create canvas for selection rectangle
    const canvas = document.createElement('canvas');
    canvas.className = 'selection-canvas';
    canvas.id = 'canvas_' + imageId;
    imageStack.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    
    // Store original image source and state
    zoomStates[imageId] = {
        originalSrc: imageSrc,
        originalMaskSrc: maskSrc,
        currentSrc: imageSrc,
        currentMaskSrc: maskSrc,
        isDrawing: false,
        startX: 0,
        startY: 0,
        zoomLevel: 1,
        fullImage: null,
        fullMask: null,
        // Track current crop region in original image coordinates
        cropX: 0,
        cropY: 0,
        cropWidth: null,  // Will be set when image loads
        cropHeight: null
    };
    
    // Preload full resolution images
    const fullImg = new Image();
    fullImg.src = imageSrc;
    fullImg.onload = () => {
        zoomStates[imageId].fullImage = fullImg;
        // Initialize crop to full image
        zoomStates[imageId].cropWidth = fullImg.naturalWidth;
        zoomStates[imageId].cropHeight = fullImg.naturalHeight;
    };
    
    if (maskSrc) {
        const fullMask = new Image();
        fullMask.src = maskSrc;
        fullMask.onload = () => {
            zoomStates[imageId].fullMask = fullMask;
        };
    }
    
    // Update canvas size to match image
    function updateCanvasSize() {
        const rect = baseImage.getBoundingClientRect();
        canvas.width = baseImage.offsetWidth;
        canvas.height = baseImage.offsetHeight;
        canvas.style.width = baseImage.offsetWidth + 'px';
        canvas.style.height = baseImage.offsetHeight + 'px';
    }
    
    baseImage.addEventListener('load', updateCanvasSize);
    window.addEventListener('resize', updateCanvasSize);
    updateCanvasSize();
    
    // Prevent default drag behavior on images
    baseImage.draggable = false;
    if (overlayImage) overlayImage.draggable = false;
    
    // Mouse event handlers for selection
    imageStack.addEventListener('mousedown', (e) => {
        e.preventDefault(); // Prevent default drag behavior
        const rect = canvas.getBoundingClientRect();
        zoomStates[imageId].isDrawing = true;
        zoomStates[imageId].startX = e.clientX - rect.left;
        zoomStates[imageId].startY = e.clientY - rect.top;
    });
    
    imageStack.addEventListener('mousemove', (e) => {
        if (!zoomStates[imageId].isDrawing) return;
        
        const rect = canvas.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;
        
        // Clear and redraw selection rectangle
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 2;
        ctx.fillStyle = 'rgba(52, 152, 219, 0.1)';
        
        const width = currentX - zoomStates[imageId].startX;
        const height = currentY - zoomStates[imageId].startY;
        
        ctx.fillRect(zoomStates[imageId].startX, zoomStates[imageId].startY, width, height);
        ctx.strokeRect(zoomStates[imageId].startX, zoomStates[imageId].startY, width, height);
    });
    
    imageStack.addEventListener('mouseup', (e) => {
        if (!zoomStates[imageId].isDrawing) return;
        
        const rect = canvas.getBoundingClientRect();
        const endX = e.clientX - rect.left;
        const endY = e.clientY - rect.top;
        
        zoomStates[imageId].isDrawing = false;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Calculate selection box
        const x1 = Math.min(zoomStates[imageId].startX, endX);
        const y1 = Math.min(zoomStates[imageId].startY, endY);
        const x2 = Math.max(zoomStates[imageId].startX, endX);
        const y2 = Math.max(zoomStates[imageId].startY, endY);
        
        const width = x2 - x1;
        const height = y2 - y1;
        
        // Only zoom if selection is large enough (minimum 20x20 pixels)
        if (width > 20 && height > 20) {
            zoomToRegion(imageId, x1, y1, width, height);
        }
    });
    
    imageStack.addEventListener('mouseleave', () => {
        if (zoomStates[imageId].isDrawing) {
            zoomStates[imageId].isDrawing = false;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    });
    
    // Home button handler
    if (homeButton) {
        homeButton.addEventListener('click', () => {
            resetZoom(imageId);
        });
    }
}

function zoomToRegion(imageId, x, y, width, height) {
    const state = zoomStates[imageId];
    const baseImage = document.getElementById('img_' + imageId);
    const overlayImage = document.getElementById('mask_' + imageId);
    const homeButton = document.getElementById('home_' + imageId);
    const zoomInfo = document.getElementById('info_' + imageId);
    
    if (!state.fullImage || !state.cropWidth || !state.cropHeight) return;
    
    // Calculate selection relative to currently displayed crop region
    // The selection (x, y, width, height) is in displayed image coordinates
    // Need to map it to the current crop region coordinates
    const scaleX = state.cropWidth / baseImage.offsetWidth;
    const scaleY = state.cropHeight / baseImage.offsetHeight;
    
    // New crop region relative to current crop
    const relativeX = Math.floor(x * scaleX);
    const relativeY = Math.floor(y * scaleY);
    const relativeWidth = Math.floor(width * scaleX);
    const relativeHeight = Math.floor(height * scaleY);
    
    // Convert to original image coordinates
    const cropX = state.cropX + relativeX;
    const cropY = state.cropY + relativeY;
    const cropWidth = relativeWidth;
    const cropHeight = relativeHeight;
    
    // Create cropped canvas for base image
    const canvas = document.createElement('canvas');
    canvas.width = cropWidth;
    canvas.height = cropHeight;
    const ctx = canvas.getContext('2d');
    
    ctx.drawImage(
        state.fullImage,
        cropX, cropY, cropWidth, cropHeight,
        0, 0, cropWidth, cropHeight
    );
    
    const croppedDataUrl = canvas.toDataURL('image/png');
    baseImage.src = croppedDataUrl;
    state.currentSrc = croppedDataUrl;
    
    // Crop mask image if it exists
    if (overlayImage && state.fullMask) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(
            state.fullMask,
            cropX, cropY, cropWidth, cropHeight,
            0, 0, cropWidth, cropHeight
        );
        
        const croppedMaskUrl = canvas.toDataURL('image/png');
        overlayImage.src = croppedMaskUrl;
        state.currentMaskSrc = croppedMaskUrl;
    }
    
    // Update crop region tracking
    state.cropX = cropX;
    state.cropY = cropY;
    state.cropWidth = cropWidth;
    state.cropHeight = cropHeight;
    
    // Update zoom level
    const zoomFactorX = state.fullImage.naturalWidth / cropWidth;
    const zoomFactorY = state.fullImage.naturalHeight / cropHeight;
    state.zoomLevel = Math.max(zoomFactorX, zoomFactorY);
    
    // Enable home button and update info
    if (homeButton) homeButton.disabled = false;
    if (zoomInfo) zoomInfo.textContent = `Zoom: ${state.zoomLevel.toFixed(1)}x`;
}

function resetZoom(imageId) {
    const state = zoomStates[imageId];
    const baseImage = document.getElementById('img_' + imageId);
    const overlayImage = document.getElementById('mask_' + imageId);
    const homeButton = document.getElementById('home_' + imageId);
    const zoomInfo = document.getElementById('info_' + imageId);
    
    // Reset to original images
    baseImage.src = state.originalSrc;
    state.currentSrc = state.originalSrc;
    
    if (overlayImage && state.originalMaskSrc) {
        overlayImage.src = state.originalMaskSrc;
        state.currentMaskSrc = state.originalMaskSrc;
    }
    
    state.zoomLevel = 1;
    
    // Reset crop region to full image
    state.cropX = 0;
    state.cropY = 0;
    if (state.fullImage) {
        state.cropWidth = state.fullImage.naturalWidth;
        state.cropHeight = state.fullImage.naturalHeight;
    }
    
    // Disable home button and clear info
    if (homeButton) homeButton.disabled = true;
    if (zoomInfo) zoomInfo.textContent = '';
}

// Load and render image data
function loadImageData(dataFile) {
    const dataUrl = `${dataFile}?v=${VIEWER_CACHE_BUST}`;
    fetch(dataUrl, { cache: 'no-store' })
        .then(response => response.json())
        .then(data => {
            renderViewer(data);
        })
        .catch(error => {
            console.error('Error loading image data:', error);
            document.getElementById('content').innerHTML = 
                '<p style="text-align: center; color: red;">Error loading image data</p>';
        });
}

function renderViewer(data) {
    const container = document.getElementById('content');
    const filterControlsHtml = `
        <div class='review-filter-controls'>
            <label class='review-toggle-label'>
                <input type='checkbox' id='review_only_toggle'>
                Filter for review
            </label>
            <label class='review-toggle-label'>
                <input type='checkbox' id='collapse_all_toggle'>
                Collapse all
            </label>
            <button type='button' id='export_reviewed_csv' class='export-review-button'>Export reviewed CSV</button>
        </div>
    `;
    container.insertAdjacentHTML('beforeend', filterControlsHtml);
    const reviewOnlyToggle = document.getElementById('review_only_toggle');
    if (reviewOnlyToggle) {
        reviewOnlyToggle.addEventListener('change', (event) => {
            setReviewFilterEnabled(Boolean(event.target.checked));
        });
    }
    const collapseAllToggle = document.getElementById('collapse_all_toggle');
    if (collapseAllToggle) {
        collapseAllToggle.addEventListener('change', (event) => {
            toggleCollapseAll(Boolean(event.target.checked));
        });
    }
    const exportReviewedCsvButton = document.getElementById('export_reviewed_csv');
    if (exportReviewedCsvButton) {
        exportReviewedCsvButton.addEventListener('click', () => {
            exportReviewedCsv();
        });
    }
    
    data.sections.forEach(section => {
        const sectionId = normalizeIdPart(section.title);
        // Section header
        const sectionHtml = `
            <div class='section-block' id='section_${sectionId}'>
                <h2 style='text-align: center; color: #2c3e50; margin-top: 40px; border-top: 3px solid #3498db; padding-top: 20px;'>
                    ${section.title}
                </h2>
                <p style='text-align: center; color: #7f8c8d;'>Directory: ${section.directory}</p>
                <p style='text-align: center; color: #7f8c8d;'>Total files: ${section.images.length}</p>
                <div class='section-rows' id='section_rows_${sectionId}'></div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', sectionHtml);
        const sectionRowsContainer = document.getElementById(`section_rows_${sectionId}`);
        
        const imagesByDataset = new Map();
        section.images.forEach(img => {
            const datasetName = img.dataset_name || 'Unknown Dataset';
            if (!imagesByDataset.has(datasetName)) {
                imagesByDataset.set(datasetName, []);
            }
            imagesByDataset.get(datasetName).push(img);
        });

        Array.from(imagesByDataset.entries()).forEach(([datasetName, datasetImages], datasetIdx) => {
            const datasetKey = `${sectionId}_${normalizeIdPart(datasetName)}_${datasetIdx + 1}`;
            const datasetHtml = `
                <div class='dataset-block' id='dataset_${datasetKey}'>
                    <div class='dataset-header'>
                        <div class='dataset-title'>${datasetName}</div>
                        <div class='dataset-controls'>
                            <label class='dataset-review-label' for='dataset_review_${datasetKey}'>
                                <input
                                    type='checkbox'
                                    id='dataset_review_${datasetKey}'
                                    class='dataset-review-checkbox'
                                    data-dataset-key='${datasetKey}'
                                    onchange='toggleDatasetReview("${datasetKey}", this.checked)'
                                >
                                Check dataset for review
                            </label>
                            <label class='dataset-collapse-label' for='dataset_collapse_${datasetKey}'>
                                <input
                                    type='checkbox'
                                    id='dataset_collapse_${datasetKey}'
                                    class='dataset-collapse-checkbox'
                                    data-dataset-key='${datasetKey}'
                                    onchange='toggleDatasetCollapse("${datasetKey}", this.checked)'
                                >
                                Collapse
                            </label>
                        </div>
                    </div>
                    <div class='dataset-rows' id='dataset_rows_${datasetKey}'></div>
                </div>
            `;
            sectionRowsContainer.insertAdjacentHTML('beforeend', datasetHtml);

            const datasetRowsContainer = document.getElementById(`dataset_rows_${datasetKey}`);
            datasetImages.forEach((img, idx) => {
                const rowId = `row_${section.title.replace(/\s+/g, '_')}_${datasetKey}_${idx + 1}`;
                renderImage(datasetRowsContainer, img, rowId, datasetKey, section.title, section.directory);
            });
        });
    });

    syncCollapseAllToggle();
    
    // Initialize zoom for all images
    setTimeout(() => {
        document.querySelectorAll('.image-stack').forEach((stack) => {
            const stackId = stack.id.replace('stack_', '');
            const baseImage = document.getElementById('img_' + stackId);
            const overlayImage = document.getElementById('mask_' + stackId);
            
            if (baseImage) {
                const imageSrc = baseImage.src;
                const maskSrc = overlayImage ? overlayImage.src : null;
                initializeZoom(stackId, imageSrc, maskSrc);
            }
        });
    }, 100);
}

function renderImage(container, imgData, rowId, datasetKey, sectionTitle, sectionDirectory) {
    const hasMask = imgData.mask_axis0 !== null;
    const fallbackStem = (imgData.filename || '').replace(/\.tiff?$/i, '').replace(/_image$/i, '');
    const displayDataset = imgData.dataset_name || 'Unknown Dataset';
    const displayStem = imgData.file_stem || fallbackStem || imgData.filename || 'Unknown File';
    const rowTitle = `${displayDataset} : ${displayStem}`;
    const fullFilePath = sectionDirectory && imgData.filename
        ? `${sectionDirectory}/${imgData.filename}`
        : (imgData.filename || '');

    rowMetadata.set(rowId, {
        section_title: sectionTitle,
        dataset_name: displayDataset,
        file_stem: displayStem,
        filename: imgData.filename || '',
        swc_file: imgData.swc_file || '',
        shape: imgData.shape || '',
        full_file_path: fullFilePath,
    });
    
    let html = `
        <div class='image-container qc-row' id='review_row_${rowId}' data-row-id='${rowId}'>
            <div class='filename'>${rowTitle}</div>
            <div class='review-row-controls'>
                <label class='review-checkbox-label' for='review_checkbox_${rowId}'>
                    <input
                        type='checkbox'
                        id='review_checkbox_${rowId}'
                        class='review-checkbox'
                        data-row-id='${rowId}'
                        data-dataset-key='${datasetKey}'
                        onchange='toggleRowReview("${rowId}", this.checked)'
                    >
                    Check for review
                </label>
            </div>
            <div class='shape-info'>Volume Shape: ${imgData.shape}${imgData.swc_file ? ` | SWC: ${imgData.swc_file}` : ' | <span style="color: orange;">No SWC file found</span>'}</div>
    `;
    
    if (hasMask) {
        html += `
            <div class='slider-container'>
                <label for='slider_${rowId}'>Neuron Mask Opacity: <span id='value_${rowId}'>0.5</span></label>
                <input type='range' min='0' max='100' value='50' class='opacity-slider' id='slider_${rowId}' 
                       oninput='updateOpacity("${rowId}", this.value)'>
            </div>
        `;
    }
    
    html += `<div class='mip-grid'>`;
    
    // Render each axis
    ['axis0', 'axis1', 'axis2'].forEach((axis, i) => {
        const imgId = `${rowId}_${axis}`;
        const imgFile = imgData[`img_${axis}`];
        const maskFile = imgData[`mask_${axis}`];
        const shape = imgData[`shape_${axis}`];
        const imgUrl = `images/${imgFile}?v=${VIEWER_CACHE_BUST}`;
        const maskUrl = maskFile ? `images/${maskFile}?v=${VIEWER_CACHE_BUST}` : null;
        
        html += `
            <div class='mip-column'>
                <div class='mip-label'>Axis ${i} Projection<br/>Shape: ${shape}</div>
                <div class='zoom-controls'>
                    <button class='home-button' id='home_${imgId}' disabled>Reset</button>
                </div>
                <div class='image-stack' id='stack_${imgId}'>
                    <img class='mip-image base-image' id='img_${imgId}' src='${imgUrl}' alt='${imgData.filename} axis ${i}' draggable='false'>
        `;
        
        if (maskFile) {
            html += `<img class='mip-image overlay-image' id='mask_${imgId}' src='${maskUrl}' alt='mask' draggable='false'>`;
        }
        
        html += `
                </div>
                <div class='zoom-info' id='info_${imgId}'></div>
            </div>
        `;
    });
    
    html += `
        </div>
    </div>`;
    
    container.insertAdjacentHTML('beforeend', html);
}

// Load data when page is ready
document.addEventListener('DOMContentLoaded', () => {
    loadImageData('image_data.json');
});
