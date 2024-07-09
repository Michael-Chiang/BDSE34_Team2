$(document).ready(function () {

    let currentPage = 1;
    let totalPages = 1;
    let resultsData = {};  // 保存请求返回的所有数据
    let priceRange = { min_price: 0, max_price: 100 };  // 初始化价格范围


    function toggleSelection(button) {
        button.classList.toggle('selected');
    }

    function updateIndustryButtons() {
        let selectedSectors = [];
        let buttons = document.querySelectorAll('.sector-button');
        buttons.forEach(button => {
            if (button.classList.contains('selected')) {
                selectedSectors.push(button.getAttribute('data-sector'));
            }
        });
        $.post('/filter_sectors', { sectors: selectedSectors.join(',') }, function (data) {
            var buttonsHtml = '';
            for (var i = 0; i < data.industries.length; i++) {
                buttonsHtml += '<li><button class="industry-button button small fit" data-industry="' + data.industries[i] + '" onclick="toggleIndustryAndFilter(this)">' + data.industries[i] + '</button></li>';
            }
            $('#industry-buttons').html(buttonsHtml);
        });
    }


    function updateResults() {
        let selectedSectors = [];
        let sectorButtons = document.querySelectorAll('.sector-button');
        sectorButtons.forEach(button => {
            if (button.classList.contains('selected')) {
                selectedSectors.push(button.getAttribute('data-sector'));
            }
        });

        let selectedIndustries = [];
        let industryButtons = document.querySelectorAll('.industry-button');
        industryButtons.forEach(button => {
            if (button.classList.contains('selected')) {
                selectedIndustries.push(button.getAttribute('data-industry'));
            }
        });

        const stockID = $('#stock-id-input').val(); // 获取输入框中的股票代码
        const minPrice = $('#price-min').val();
        const maxPrice = $('#price-max').val();

        $.post('/filter_results', {
            sectors: selectedSectors.join(','),
            industries: selectedIndustries.join(','),
            stock_id: stockID,
            min_price: minPrice,
            max_price: maxPrice
        }, function (data) {
            if (data && data.results && data.columns) {
                resultsData = data;
                totalPages = Math.ceil(data.results.length / 20);
                displayResults();
                updatePagination();
                $('#total-count').text(`Total results: ${data.results.length}`);  // 显示总记录数
                updatePriceRange(data.price_range);  // 更新价格范围
            } else {
                $('#results').html('<p>No results found</p>');
                $('#total-count').text(`Total results: 0`);  // 显示总记录数


            }
        });
    }

    $('#price-min, #price-max').on('slidestop', function () {
        updateResults();
    });

    function displayResults() {
        const { results, columns } = resultsData;
        var table = '<table border="1"><thead><tr>';
        for (var i = 0; i < columns.length; i++) {
            table += '<th>' + columns[i] + '</th>';
        }
        table += '</tr></thead><tbody>';
        let start = (currentPage - 1) * 20;
        let end = start + 20;
        let pageResults = results.slice(start, end);
        for (var i = 0; i < pageResults.length; i++) {
            table += '<tr>';
            for (var j = 0; j < columns.length; j++) {
                if (columns[j] === 'Symbol') {
                    table += '<td><a href="/stock/' + pageResults[i][columns[j]] + '">' + pageResults[i][columns[j]] + '</a></td>';
                } else {
                    table += '<td>' + pageResults[i][columns[j]] + '</td>';
                }
            }
            table += '</tr>';
        }
        table += '</tbody></table>';
        $('#results').html(table);
    }

    function updatePagination() {
        let paginationHtml = '';

        paginationHtml += `<li><button class="button small ${currentPage === 1 ? 'disabled' : ''}" id="prev-button">Prev</button></li>`;

        if (currentPage > 3) {
            paginationHtml += `<li><button class="page" data-page="1">1</button></li>`;
            if (currentPage > 4) {
                paginationHtml += `<li><span>...</span></li>`;
            }
        }

        for (let i = Math.max(1, currentPage - 2); i <= Math.min(totalPages, currentPage + 2); i++) {
            paginationHtml += `<li><button class="page ${i === currentPage ? 'active' : ''}" data-page="${i}">${i}</button></li>`;
        }

        if (currentPage < totalPages - 2) {
            if (currentPage < totalPages - 3) {
                paginationHtml += `<li><span>...</span></li>`;
            }
            paginationHtml += `<li><button class="page" data-page="${totalPages}">${totalPages}</button></li>`;
        }

        paginationHtml += `<li><button class="button small ${currentPage === totalPages ? 'disabled' : ''}" id="next-button">Next</button></li>`;
        $('#pagination-buttons').html(paginationHtml);

        // 绑定上一页和下一页按钮的事件
        $('#prev-button').off('click').on('click', function () {
            if (currentPage > 1) {
                goToPage(currentPage - 1);
            }
        });

        $('#next-button').off('click').on('click', function () {
            if (currentPage < totalPages) {
                goToPage(currentPage + 1);
            }
        });

        // 绑定具体页码按钮的事件
        $('.page').off('click').on('click', function () {
            const page = $(this).data('page');
            if (page) {
                goToPage(page);
            }
        });

        // 绑定Go按钮的事件
        $('#goto-page-button').off('click').on('click', function () {
            const page = parseInt($('#goto-page-input').val());
            if (page) {
                goToPage(page);
            }
        });

        // 调整小屏幕上的分页显示
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

    updateResults();


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

    function updatePriceRange(price_range) {
        priceRange = price_range;  // 更新全局的价格范围
        $('#minLabel').text(price_range.min_price);
        $('#maxLabel').text(price_range.max_price);
        $('#price-min').attr('min', price_range.min_price).attr('max', price_range.max_price).val(price_range.min_price);
        $('#price-max').attr('min', price_range.min_price).attr('max', price_range.max_price).val(price_range.max_price);
        $('#price-min, #price-max').slider('refresh');
    }

    $('#price-min, #price-max').on('slidestop', function () {
        updateResults();
    });

    window.updateResults = updateResults;



    // // 使用iframe引入jQuery Mobile样式
    // const iframe = document.createElement("iframe");
    // iframe.style.display = "none";
    // document.body.appendChild(iframe);
    // const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
    // const script = iframeDoc.createElement("script");
    // script.src = "https://code.jquery.com/jquery-1.11.3.min.js";
    // script.onload = function () {
    //     const mobileScript = iframeDoc.createElement("script");
    //     mobileScript.src = "https://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.js";
    //     mobileScript.onload = function () {
    //         const fieldContain = iframeDoc.createElement("div");
    //         fieldContain.setAttribute("data-role", "fieldcontain");
    //         fieldContain.innerHTML = `
    //             <label for="price-range">价格范围:</label>
    //             <input type="range" name="price-min" id="price-min" min="${$('#price-min').attr('min')}" max="${$('#price-max').attr('max')}" value="${$('#price-min').val()}">
    //             <input type="range" name="price-max" id="price-max" min="${$('#price-min').attr('min')}" max="${$('#price-max').attr('max')}" value="${$('#price-max').val()}">
    //         `;
    //         iframeDoc.body.appendChild(fieldContain);
    //         $('#price-slider-container').html(iframeDoc.body.innerHTML);

    //         $('#price-min, #price-max').slider('refresh');
    //     };
    //     iframeDoc.body.appendChild(mobileScript);
    // };
    // iframeDoc.body.appendChild(script);

});

