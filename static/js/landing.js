// ==========================================
// LANDING PAGE INTERACTIVE JAVASCRIPT
// ==========================================

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    
    // ==========================================
    // SCROLL REVEAL ANIMATION SYSTEM
    // ==========================================
    
    // Select all elements with reveal-on-scroll class
    const revealElements = document.querySelectorAll('.reveal-on-scroll');
    
    // Create Intersection Observer for scroll reveal
    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Add revealed class when element is in viewport
                entry.target.classList.add('revealed');
                
                // Optional: Stop observing after reveal (performance optimization)
                // revealObserver.unobserve(entry.target);
            }
        });
    }, {
        root: null, // Use viewport as root
        rootMargin: '0px',
        threshold: 0.15 // Trigger when 15% of element is visible
    });
    
    // Observe all reveal elements
    revealElements.forEach(element => {
        revealObserver.observe(element);
    });
    
    // ==========================================
    // NAVBAR SCROLL BEHAVIOR
    // ==========================================
    
    const navbar = document.getElementById('navbar');
    let lastScrollY = window.scrollY;
    
    // Add/remove scrolled class based on scroll position
    function handleNavbarScroll() {
        const currentScrollY = window.scrollY;
        
        if (currentScrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
        
        lastScrollY = currentScrollY;
    }
    
    // Listen for scroll events (with performance optimization)
    let ticking = false;
    
    window.addEventListener('scroll', function() {
        if (!ticking) {
            window.requestAnimationFrame(function() {
                handleNavbarScroll();
                ticking = false;
            });
            ticking = true;
        }
    });
    
    // Initial check
    handleNavbarScroll();
    
    // ==========================================
    // SMOOTH SCROLLING FOR ANCHOR LINKS
    // ==========================================
    
    // Get all navigation links with hash anchors
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Only prevent default for same-page anchors
            if (this.getAttribute('href').startsWith('#')) {
                e.preventDefault();
                
                const targetId = this.getAttribute('href');
                const targetElement = document.querySelector(targetId);
                
                if (targetElement) {
                    // Calculate offset for fixed navbar
                    const navbarHeight = navbar.offsetHeight;
                    const targetPosition = targetElement.offsetTop - navbarHeight - 20;
                    
                    // Smooth scroll to target
                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                    
                    // Close mobile menu if open
                    const navLinks = document.querySelector('.nav-links');
                    if (navLinks.classList.contains('active')) {
                        navLinks.classList.remove('active');
                    }
                }
            }
        });
    });
    
    // ==========================================
    // MOBILE NAVIGATION TOGGLE
    // ==========================================
    
    const navToggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');
    
    if (navToggle && navLinks) {
        navToggle.addEventListener('click', function() {
            navLinks.classList.toggle('active');
            
            // Animate hamburger menu icon (optional)
            const spans = this.querySelectorAll('span');
            if (navLinks.classList.contains('active')) {
                // Transform to X
                spans[0].style.transform = 'rotate(45deg) translate(5px, 5px)';
                spans[1].style.opacity = '0';
                spans[2].style.transform = 'rotate(-45deg) translate(7px, -6px)';
            } else {
                // Reset to hamburger
                spans[0].style.transform = 'none';
                spans[1].style.opacity = '1';
                spans[2].style.transform = 'none';
            }
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', function(e) {
            if (!navToggle.contains(e.target) && !navLinks.contains(e.target)) {
                navLinks.classList.remove('active');
                const spans = navToggle.querySelectorAll('span');
                spans[0].style.transform = 'none';
                spans[1].style.opacity = '1';
                spans[2].style.transform = 'none';
            }
        });
    }
    
    // ==========================================
    // INTERACTIVE CARD HOVER EFFECTS
    // ==========================================
    
    // Add subtle parallax effect to feature cards on hover
    const cards = document.querySelectorAll('.feature-card, .pipeline-step, .difference-card');
    
    cards.forEach(card => {
        card.addEventListener('mousemove', function(e) {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateX = (y - centerY) / 20;
            const rotateY = (centerX - x) / 20;
            
            // Apply subtle 3D effect
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-8px)`;
        });
        
        card.addEventListener('mouseleave', function() {
            // Reset transform on mouse leave
            card.style.transform = '';
        });
    });
    
    // ==========================================
    // ANIMATED COUNTER FOR HERO STATS
    // ==========================================
    
    // Function to animate counting numbers
    function animateCounter(element, target, duration = 2000) {
        const start = 0;
        const startTime = performance.now();
        
        function updateCounter(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function for smooth animation
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = Math.floor(start + (target - start) * easeOut);
            
            element.textContent = current;
            
            if (progress < 1) {
                requestAnimationFrame(updateCounter);
            } else {
                element.textContent = target;
            }
        }
        
        requestAnimationFrame(updateCounter);
    }
    
    // Observe hero stats for animation
    const heroStats = document.querySelectorAll('.stat-number');
    
    const statsObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.classList.contains('animated')) {
                entry.target.classList.add('animated');
                const text = entry.target.textContent;
                
                // Animate if it's a number
                const number = parseInt(text);
                if (!isNaN(number)) {
                    animateCounter(entry.target, number);
                }
                
                statsObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    heroStats.forEach(stat => {
        statsObserver.observe(stat);
    });
    
    // ==========================================
    // BUTTON CLICK ANIMATIONS
    // ==========================================
    
    const buttons = document.querySelectorAll('.btn');
    
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            // Create ripple effect
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');
            
            this.appendChild(ripple);
            
            // Remove ripple after animation
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
    
    // ==========================================
    // DYNAMIC BACKGROUND PARTICLES (Optional)
    // ==========================================
    
    // Lightweight particle animation for hero section
    const heroSection = document.querySelector('.hero');
    
    if (heroSection && window.innerWidth > 768) {
        // Only add particles on desktop for performance
        createParticles(heroSection);
    }
    
    function createParticles(container) {
        const particleCount = 30;
        const particles = [];
        
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.classList.add('particle');
            
            // Random size
            const size = Math.random() * 4 + 2;
            particle.style.width = size + 'px';
            particle.style.height = size + 'px';
            
            // Random position
            particle.style.position = 'absolute';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            
            // Random opacity
            particle.style.opacity = Math.random() * 0.5 + 0.1;
            
            // Random color
            const colors = ['#6C63FF', '#4A90E2', '#FF6B9D', '#00D9FF'];
            particle.style.background = colors[Math.floor(Math.random() * colors.length)];
            particle.style.borderRadius = '50%';
            
            // Animation
            particle.style.animation = `float ${Math.random() * 10 + 10}s ease-in-out infinite`;
            particle.style.animationDelay = Math.random() * 5 + 's';
            
            container.appendChild(particle);
        }
        
        // Add floating animation
        if (!document.getElementById('particle-animation')) {
            const style = document.createElement('style');
            style.id = 'particle-animation';
            style.textContent = `
                @keyframes float {
                    0%, 100% {
                        transform: translate(0, 0) scale(1);
                    }
                    33% {
                        transform: translate(30px, -30px) scale(1.2);
                    }
                    66% {
                        transform: translate(-20px, 20px) scale(0.8);
                    }
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    // ==========================================
    // SCROLL PROGRESS INDICATOR (Optional)
    // ==========================================
    
    // Create progress bar at top of page
    const progressBar = document.createElement('div');
    progressBar.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        height: 3px;
        background: linear-gradient(135deg, #6C63FF 0%, #4A90E2 100%);
        z-index: 10000;
        transition: width 0.1s linear;
    `;
    document.body.appendChild(progressBar);
    
    window.addEventListener('scroll', function() {
        const windowHeight = document.documentElement.scrollHeight - window.innerHeight;
        const scrolled = (window.scrollY / windowHeight) * 100;
        progressBar.style.width = scrolled + '%';
    });
    
    // ==========================================
    // LAZY LOADING FOR PERFORMANCE
    // ==========================================
    
    // Lazy load images and heavy elements
    const lazyElements = document.querySelectorAll('[data-lazy]');
    
    if (lazyElements.length > 0) {
        const lazyObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const element = entry.target;
                    
                    // Load data
                    if (element.dataset.src) {
                        element.src = element.dataset.src;
                    }
                    
                    lazyObserver.unobserve(element);
                }
            });
        }, { threshold: 0.1 });
        
        lazyElements.forEach(element => {
            lazyObserver.observe(element);
        });
    }
    
    // ==========================================
    // ACCESSIBILITY ENHANCEMENTS
    // ==========================================
    
    // Skip to main content link (for keyboard users)
    const skipLink = document.createElement('a');
    skipLink.href = '#main-content';
    skipLink.textContent = 'Skip to main content';
    skipLink.style.cssText = `
        position: absolute;
        top: -40px;
        left: 0;
        background: #6C63FF;
        color: white;
        padding: 8px 16px;
        z-index: 100000;
        transition: top 0.3s;
    `;
    
    skipLink.addEventListener('focus', function() {
        this.style.top = '0';
    });
    
    skipLink.addEventListener('blur', function() {
        this.style.top = '-40px';
    });
    
    document.body.appendChild(skipLink);
    
    // ==========================================
    // FORMATTED NUMBER DISPLAY
    // ==========================================
    
    // Format large numbers with K, M, B suffixes
    function formatNumber(num) {
        if (num >= 1000000000) {
            return (num / 1000000000).toFixed(1) + 'B';
        }
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        }
        if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }
    
    // ==========================================
    // CONSOLE WELCOME MESSAGE
    // ==========================================
    
    console.log('%c👋 Welcome to Hybrid Data Analytics System!', 
        'color: #6C63FF; font-size: 20px; font-weight: bold;');
    console.log('%c📊 Turn your raw data into smart business decisions', 
        'color: #4A90E2; font-size: 14px;');
    console.log('%c💡 Built with modern web technologies', 
        'color: #00D9FF; font-size: 12px;');
    
});

// ==========================================
// PERFORMANCE OPTIMIZATION
// ==========================================

// Debounce function for scroll events
function debounce(func, wait = 100) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle function for animations
function throttle(func, limit = 1000 / 60) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// ==========================================
// ERROR HANDLING
// ==========================================

window.addEventListener('error', function(e) {
    console.error('An error occurred:', e.error);
    // In production, you might want to send this to an error tracking service
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
});
