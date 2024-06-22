$(document).ready(function () {
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
                buttonsHtml += '<button class="industry-button" data-industry="' + data.industries[i] + '" onclick="toggleIndustryAndFilter(this)">' + data.industries[i] + '</button>';
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

        $.post('/filter_results', { sectors: selectedSectors.join(','), industries: selectedIndustries.join(',') }, function (data) {
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
        });
    }

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
