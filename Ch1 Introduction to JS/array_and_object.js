let arr = [1, 2, 3, 4, 5];

for (let i = 0; i < 5; i++) {
  arr.push(10); // O(1)
  arr.unshift(10); // O(n)
}

console.log(arr);
