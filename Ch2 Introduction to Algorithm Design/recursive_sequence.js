function rs(n) {
  console.log(`We are inside function rs(${n})`);
  if (n == 1) {
    return 10;
  } else {
    return rs(n - 1) + 15;
  }
}

console.log(rs(3));
