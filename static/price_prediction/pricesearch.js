let data;

function loadCSV() {
    fetch('/prices/data/')
    .then(response => response.json())
    .then(csvData => {
        data = csvData;
        console.log(data);
    });
}

function searchCommodity() {
    const searchValue = document.getElementById('searchInput').value.toLowerCase();
    const dateValue = document.getElementById('dateInput').value;
    console.log(searchValue);
    console.log(dateValue);
    if (!data) {
        console.error('CSV data not loaded');
        return;
    }
    const result = data.filter(row => {
        const commodity = row['Commodity'].toLowerCase();
        return commodity.includes(searchValue) && row['Date'] === dateValue;
    });
    displayResult(result);
}

function displayResult(result) {
    let html = '<table border="1"><tr><th>Commodity</th><th>Date</th><th>Unit</th><th>Minimum</th><th>Maximum</th><th>Average</th></tr>';
    result.forEach(row => {
        html += `<tr><td>${row['Commodity']}</td><td>${row['Date']}</td><td>${row['Unit']}</td><td>${row['Minimum']}</td><td>${row['Maximum']}</td><td>${row['Average']}</td></tr>`;
    });
    html += '</table>';
    document.getElementById('result').innerHTML = html;
}

window.onload = function() {
    loadCSV();
};
