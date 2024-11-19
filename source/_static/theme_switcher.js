document.addEventListener("DOMContentLoaded", function () {
    const darkModeMediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const logoElement = document.querySelector(".logo img");

    function updateLogo() {
        if (darkModeMediaQuery.matches) {
            logoElement.src = "../assets/figs/logo_dark.png";
        } else {
            logoElement.src = "../assets/figs/logo_light.png";
        }
    }

    // 初始化
    updateLogo();

    // 当主题切换时更新
    darkModeMediaQuery.addEventListener("change", updateLogo);
});

