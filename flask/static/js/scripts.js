$(document).ready(function () {

    let currentPage = 1;
    let totalPages = 1;
    let allResults = [];
    let columns = [];

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

        $.post('/filter_results', {
            sectors: selectedSectors.join(','),
            industries: selectedIndustries.join(','),

        }, function (data) {
            //console.log(data);
            allResults = data.results;  // 保存所有结果
            columns = data.columns;  // 保存列信息
            data = data
            totalPages = Math.ceil(allResults.length / 20);  // 计算总页数
            displayResults(data, allResults, columns);  // 显示当前页结果
            updatePagination();  // 更新分页按钮
            // var table = '<table border="1"><thead><tr>';
            // for (var i = 0; i < data.columns.length; i++) {
            //     table += '<th>' + data.columns[i] + '</th>';
            // }
            // table += '</tr></thead><tbody>';
            // for (var i = 0; i < data.results.length; i++) {
            //     table += '<tr>';
            //     for (var j = 0; j < data.columns.length; j++) {
            //         if (data.columns[j] === 'Symbol') {
            //             table += '<td><a href="/stock/' + data.results[i][data.columns[j]] + '">' + data.results[i][data.columns[j]] + '</a></td>';
            //         } else {
            //             table += '<td>' + data.results[i][data.columns[j]] + '</td>';
            //         }
            //     }
            //     table += '</tr>';
            // }
            // table += '</tbody></table>';
            // $('#results').html(table);

            // totalPages = data.total_pages;  // 更新总页数
            // updatePagination();  // 更新分页按钮
        });
    }

    function displayResults(data, allResults, columns) {
        //console.log(data);

        var table = '<table border="1"><thead><tr>';
        for (var i = 0; i < data.columns.length; i++) {
            table += '<th>' + data.columns[i] + '</th>';
        }
        table += '</tr></thead><tbody>';

        let start = (currentPage - 1) * 20;
        let end = start + 20;
        let pageResults = allResults.slice(start, end);  // 获取当前页的数据





        for (var i = 0; i < pageResults.length; i++) {
            table += '<tr>';
            for (var j = 0; j < data.columns.length; j++) {
                if (columns[j] === 'Symbol') {
                    table += '<td><a href="/stock/' + pageResults[i][columns[j]] + '">' + pageResults[i][data.columns[j]] + '</a></td>';
                } else {
                    table += '<td>' + pageResults[i][data.columns[j]] + '</td>';
                }
            }
            table += '</tr>';
        }
        table += '</tbody></table>';
        $('#results').html(table);

        totalPages = data.total_pages;  // 更新总页数
        updatePagination();  // 更新分页按钮
    }

    function updatePagination() {
        let paginationHtml = '';
        for (let i = 1; i <= totalPages; i++) {
            paginationHtml += `<li><a href="#" class="page ${i === currentPage ? 'active' : ''}" onclick="goToPage(${i})">${i}</a></li>`;
        }
        $('#pagination-buttons').html(paginationHtml);
        document.getElementById('prev-button').classList.toggle('disabled', currentPage === 1);
        document.getElementById('next-button').classList.toggle('disabled', currentPage === totalPages);
    }

    function goToPage(page) {
        if (page < 1 || page > totalPages) return;
        currentPage = page;
        displayResults(data, allResults, columns);  // 显示当前页结果
    }

    document.getElementById('prev-button').addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            displayResults(data, allResults, columns);  // 显示当前页结果
        }
    });

    document.getElementById('next-button').addEventListener('click', () => {
        if (currentPage < totalPages) {
            currentPage++;
            displayResults(data, allResults, columns);  // 显示当前页结果
        }
    });

    // 初始加载数据
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
});
