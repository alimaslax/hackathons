#include <SFML/Graphics.hpp>

#include <cmath>

int main()
{
    // Create a borderless window
    sf::RenderWindow window(sf::VideoMode(800, 600), "Basketball Dribbling", sf::Style::None);
    window.setFramerateLimit(60);

    // Enable per-pixel alpha values
    window.create(sf::VideoMode(800, 600), "Basketball Dribbling", sf::Style::None);

    // Make the window's background transparent
    window.setPosition(sf::Vector2i(100, 100)); // Position the window on the screen

    sf::CircleShape basketball(30.f);
    basketball.setFillColor(sf::Color(255, 140, 0));
    basketball.setOrigin(30.f, 30.f);

    float ballX     = 400.f;
    float ballY     = 300.f;
    float velocity  = 0.f;
    float gravity   = 0.5f;
    float dampening = 0.8f;

    bool isMousePressed = false;

    sf::Clock clock;

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
            else if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left)
                isMousePressed = true;
            else if (event.type == sf::Event::MouseButtonReleased && event.mouseButton.button == sf::Mouse::Left)
                isMousePressed = false;
            else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)
                window.close();
        }

        float deltaTime = clock.restart().asSeconds();

        if (isMousePressed)
        {
            sf::Vector2i mousePosition = sf::Mouse::getPosition(window);
            ballX                      = static_cast<float>(mousePosition.x);
            ballY                      = static_cast<float>(mousePosition.y);
            velocity                   = 0.f;
        }
        else
        {
            velocity += gravity;
            ballY += velocity;

            if (ballY + 30.f > 570.f)
            {
                ballY    = 570.f - 30.f;
                velocity = -velocity * dampening;
            }
        }

        basketball.setPosition(ballX, ballY);

        window.clear(sf::Color::Transparent);
        window.draw(basketball);
        window.display();
    }

    return 0;
}