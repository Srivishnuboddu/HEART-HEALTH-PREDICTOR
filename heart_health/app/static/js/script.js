// Optional JavaScript code for interactions and dynamic effects
document.addEventListener("DOMContentLoaded", function () {
    // Example: Animate elements on scroll or add more effects
    let elements = document.querySelectorAll('.animate-on-scroll');
    window.addEventListener('scroll', function () {
        elements.forEach(function (el) {
            let position = el.getBoundingClientRect().top;
            if (position < window.innerHeight) {
                el.classList.add('visible');
            }
        });
    });
});
