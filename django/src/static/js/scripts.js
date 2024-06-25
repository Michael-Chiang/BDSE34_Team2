$(document).ready(function () {
    function getCookie(name) {
        console.log('Getting CSRF Token'); // 调试信息
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    const csrftoken = getCookie('csrftoken');
    console.log('CSRF Token:', csrftoken); // 添加调试信息

    // 设置 AJAX 请求的 CSRF 令牌
    $.ajaxSetup({
        beforeSend: function (xhr, settings) {
            if (!/^(GET|HEAD|OPTIONS|TRACE)$/.test(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
                console.log('Setting CSRF Token:', csrftoken); // 添加调试信息
            }
        }
    });


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
        $.ajax({
            type: "POST",
            url: "/filter_sectors",
            data: { sectors: selectedSectors.join(',') },
            success: function (data) {
                console.log('Update Industry Buttons Success:', data); // 添加调试信息
                var buttonsHtml = '';
                for (var i = 0; i < data.industries.length; i++) {
                    buttonsHtml += '<button class="industry-button" data-industry="' + data.industries[i] + '" onclick="toggleIndustryAndFilter(this)">' + data.industries[i] + '</button>';
                }
                $('#industry-buttons').html(buttonsHtml);
            }
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

        $.ajax({
            type: "POST",
            url: "/filter_results",
            data: { sectors: selectedSectors.join(','), industries: selectedIndustries.join(',') },
            success: function (data) {
                var table = '<table border="1"><thead><tr>';
                for (var i = 0; i < data.columns.length; i++) {
                    table += '<th>' + data.columns[i] + '</th>';
                }
                table += '</tr></thead><tbody>';
                for (var i = 0; i < data.results.length; i++) {
                    table += '<tr>';
                    for (var j = 0; j < data.columns.length; j++) {
                        if (data.columns[j] === 'Symbol') {
                            table += '<td><a href="/stock/' + data.results[i][data.columns[j]] + '">' + data.results[i][data.columns[j]] + '</a></td>';
                        } else {
                            table += '<td>' + data.results[i][data.columns[j]] + '</td>';
                        }
                    }
                    table += '</tr>';
                }
                table += '</tbody></table>';
                $('#results').html(table);
            }
        });
    }

    window.toggleSectorAndFilter = function (button) {
        toggleSelection(button);
        updateIndustryButtons();
        updateResults();
    }

    window.toggleIndustryAndFilter = function (button) {
        toggleSelection(button);
        updateResults();
    }
});
