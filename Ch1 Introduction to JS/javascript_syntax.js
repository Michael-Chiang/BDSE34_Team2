// array
let arr = ["Harry", "Ron", "Snap"];

// JS for loop
for (let i = 0; i < arr.length; i++) {
  console.log(arr[i]);
  console.log(i);
}

// forEach
arr.forEach((person, index) => {
  console.log(person);
  console.log(index);
});

console.log("----------loop end----------");

// object
let Harry = {
  name: "Harry Potter",
  age: 40,
  married: true,
  sayHi() {
    console.log("Harry say hi to you.");
  },
};

console.log(Harry.name);
console.log(Harry["name"]);

Harry.sayHi();

console.log("----------object end----------");

// function
function add(n1, n2) {
  console.log(n1 + n2);
}

function add1(n1, n2) {
  return n1 + n2;
}

add(8, -4);

console.log(add1(10, 15));

console.log("----------function end----------");

// class
class Circle {
  constructor(radius) {
    this.radius = radius;
  }
  getArea() {
    return this.radius * this.radius * Math.PI;
  }
}

let c1 = new Circle(5);
console.log(c1.radius);
console.log(c1.getArea());

console.log("----------class end----------");
