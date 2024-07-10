$(document).ready(function () {
    let currentPage = 1;
    let totalPages = 1;
    let resultsData = {};  // 保存请求返回的所有数据
    let sliderInitialized = false;
    let minPrice = 0;
    let maxPrice = 0;

    // 切换按钮选中状态
    function toggleSelection(button) {
        button.classList.toggle('selected');
    }

    // 更新行业按钮
    function updateIndustryButtons() {
        let selectedSectors = getSelectedItems('.sector-button', 'data-sector');
        $.post('/filter_sectors', { sectors: selectedSectors.join(',') }, function (data) {
            let buttonsHtml = data.industries.map(industry => `<li><button class="industry-button button small fit" data-industry="${industry}" onclick="toggleIndustryAndFilter(this)">${industry}</button></li>`).join('');
            $('#industry-buttons').html(buttonsHtml);
        });
    }

    // 获取选中的按钮数据
    function getSelectedItems(selector, dataAttribute) {
        let selectedItems = [];
        document.querySelectorAll(selector).forEach(button => {
            if (button.classList.contains('selected')) {
                selectedItems.push(button.getAttribute(dataAttribute));
            }
        });
        return selectedItems;
    }

    // 更新结果
    function updateResults() {
        let selectedSectors = getSelectedItems('.sector-button', 'data-sector');
        let selectedIndustries = getSelectedItems('.industry-button', 'data-industry');
        const stockID = $('#stock-id-input').val(); // 获取输入框中的股票代码

        // 获取滑动条的值
        const priceRange = [minPrice, maxPrice];

        $.post('/filter_results', {
            sectors: selectedSectors.join(','),
            industries: selectedIndustries.join(','),
            stock_id: stockID,
            min_price: priceRange[0],
            max_price: priceRange[1]
        }, function (data) {
            if (data && data.results && data.columns) {
                resultsData = data;
                totalPages = Math.ceil(data.results.length / 20);
                displayResults();
                updatePagination();
                $('#total-count').text(`Total results: ${data.results.length}`);  // 显示总记录数
            } else {
                $('#results').html('<p>No results found</p>');
                $('#total-count').text(`Total results: 0`);  // 显示总记录数
            }
        });
    }

    // 显示结果
    function displayResults() {
        const { results, columns } = resultsData;
        let table = `<table border="1"><thead><tr>${columns.map(col => `<th>${col}</th>`).join('')}</tr></thead><tbody>`;
        let pageResults = results.slice((currentPage - 1) * 20, currentPage * 20);
        pageResults.forEach(result => {
            table += '<tr>' + columns.map(col => `<td>${col === 'Symbol' ? `<a href="/stock/${result[col]}">${result[col]}</a>` : result[col]}</td>`).join('') + '</tr>';
        });
        table += '</tbody></table>';
        $('#results').html(table);
    }

    // 更新分页
    function updatePagination() {
        let paginationHtml = `<li><button class="button small ${currentPage === 1 ? 'disabled' : ''}" id="prev-button">Prev</button></li>`;
        paginationHtml += createPaginationButtons();
        paginationHtml += `<li><button class="button small ${currentPage === totalPages ? 'disabled' : ''}" id="next-button">Next</button></li>`;
        $('#pagination-buttons').html(paginationHtml);

        // 绑定分页按钮事件
        bindPaginationEvents();
    }

    // 创建分页按钮
    function createPaginationButtons() {
        let buttons = '';
        if (currentPage > 3) buttons += `<li><button class="page" data-page="1">1</button></li><li><span>...</span></li>`;
        for (let i = Math.max(1, currentPage - 2); i <= Math.min(totalPages, currentPage + 2); i++) {
            buttons += `<li><button class="page ${i === currentPage ? 'active' : ''}" data-page="${i}">${i}</button></li>`;
        }
        if (currentPage < totalPages - 2) buttons += `<li><span>...</span></li><li><button class="page" data-page="${totalPages}">${totalPages}</button></li>`;
        return buttons;
    }

    // 绑定分页按钮事件
    function bindPaginationEvents() {
        $('#prev-button').off('click').on('click', function () {
            if (currentPage > 1) goToPage(currentPage - 1);
        });
        $('#next-button').off('click').on('click', function () {
            if (currentPage < totalPages) goToPage(currentPage + 1);
        });
        $('.page').off('click').on('click', function () {
            const page = $(this).data('page');
            if (page) goToPage(page);
        });
        $('#goto-page-button').off('click').on('click', function () {
            const page = parseInt($('#goto-page-input').val());
            if (page) goToPage(page);
        });
        adjustPaginationForSmallScreens();
    }

    // 调整小屏幕上的分页显示
    function adjustPaginationForSmallScreens() {
        if (window.matchMedia("(max-width: 736px)").matches) {
            $('.page').each(function () {
                const page = parseInt($(this).data('page'));
                if (Math.abs(page - currentPage) > 1 && page !== 1 && page !== totalPages) {
                    $(this).hide();
                } else {
                    $(this).show();
                }
            });
            $('.page-dots').hide();
        }
    }

    function goToPage(page) {
        if (page < 1 || page > totalPages) return;
        currentPage = page;
        displayResults();
        updatePagination();
        scrollToResults();  // 滚动到结果表格
    }

    function scrollToResults() {
        const resultsElement = document.getElementById('stock-results');
        if (resultsElement) {
            resultsElement.scrollIntoView({ behavior: 'smooth' });
        }
    }

    // 将goToPage函数绑定到window对象上
    window.goToPage = goToPage;

    function toggleSectorAndFilter(button) {
        toggleSelection(button);
        updateIndustryButtons();
        updateResults();
    }

    function toggleIndustryAndFilter(button) {
        toggleSelection(button);
        updateResults();
    }

    window.toggleSectorAndFilter = toggleSectorAndFilter;
    window.toggleIndustryAndFilter = toggleIndustryAndFilter;

    // 添加处理stockID搜索的逻辑
    $('#stock-id-search-button').on('click', function () {
        currentPage = 1;  // 重置为第一页
        updateResults();  // 更新结果
    });

    // 获取价格范围并初始化滑动条
    $.ajax({
        url: '/get_price_range',
        method: 'GET',
        success: function (data) {
            var sliderElement = document.getElementById('slider');
            minPrice = parseFloat(data.min_price);
            maxPrice = parseFloat(data.max_price);

            noUiSlider.create(sliderElement, {
                start: [minPrice, maxPrice], // 初始值
                connect: true,
                range: {
                    'min': minPrice,
                    'max': maxPrice
                },
                tooltips: [true, true]
            });

            let minValue = document.getElementById('slider-min');
            let maxValue = document.getElementById('slider-max');

            sliderElement.noUiSlider.on('update', function (values, handle) {
                if (handle === 0) {
                    minValue.innerHTML = values[0];
                    minPrice = values[0]; // 更新全局变量
                } else {
                    maxValue.innerHTML = values[1];
                    maxPrice = values[1]; // 更新全局变量
                }
            });


            sliderElement.noUiSlider.on('change', function () {
                currentPage = 1;  // 重置为第一页
                updateResults();
            });

            sliderInitialized = true;
            slider = sliderElement;
            updateResults();
        },
        error: function (error) {
            console.error('Error fetching price range:', error);
        }
    });

    // 初始化行业按钮和结果
    updateIndustryButtons();
    updateResults();
});
